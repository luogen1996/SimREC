# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import torch
import torch.nn as nn
from torch.backends import cudnn

from utils.parse_darknet_weights import parse_yolo_weights
from layers.conv_layer import *
from torchvision.models.resnet import resnet34,resnet101,resnet18
from utils.ResNetD import ResNetV1c

class CspDarkNet(nn.Module):
    def __init__(self, __C,multi_scale_outputs=False):
        super().__init__()
        self.model = nn.Sequential(
            Conv(c1=3, c2=64, k=6,s=2,p=2),
            # i = 0 ch = [64]
            Conv(c1=64, c2=128, k=3, s=2),
            # i = 1 ch = [64,128]
            C3(c1=128, c2=128, n=3),
            # i = 2 ch =[64,128,128]
            Conv(c1=128, c2=256, k=3, s=2),
            # i = 3 ch =[64,128,128,256]
            C3(c1=256, c2=256, n=6),
            # i = 4 ch =[64,128,128,256,256]
            Conv(c1=256, c2=512, k=3, s=2),
            # i = 5 ch =[64,128,128,256,256,512]
            C3(c1=512, c2=512, n=9),
            # i = 6 ch =[64,128,128,256,256,512,512]
            Conv(c1=512, c2=1024, k=3, s=2),
            # i = 7 ch =[64,128,128,256,256,512,512,1024]
            C3(c1=1024, c2=1024, n=3),
            # i = 8 ch =[64,128,128,256,256,512,512,1024,1024]
            SPPF(c1=1024, c2=1024,k=5),
            # i = 9 ch =[64,128,128,256,256,512,512,1024,1024,1024]
        )
        self.multi_scale_outputs=multi_scale_outputs
        if __C.VIS_PRETRAIN:
            self.weight_dict = torch.load(__C.PRETRAIN_WEIGHT)
            self.load_state_dict(self.weight_dict, strict=False)
    def frozen(self,module):
            if getattr(module,'module',False):
                for child in module.module():
                    for param in child.parameters():
                        param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        outputs=[]
        for i,module in enumerate(self.model):
            x=module(x)
            if i in [4, 6]:
                outputs.append(x)
        outputs.append(x)
        if self.multi_scale_outputs:
            return outputs
        else:
            return x


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class DarkNet53(nn.Module):
    def __init__(self, __C,multi_scale_outputs=False):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.module_list.append(darknet_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
        self.module_list.append(darknet_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
        self.module_list.append(darknetblock(ch=64))
        self.module_list.append(darknet_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
        self.module_list.append(darknetblock(ch=128, nblocks=2))
        self.module_list.append(darknet_conv(in_ch=128, out_ch=256, ksize=3, stride=2))
        self.module_list.append(darknetblock(ch=256, nblocks=8))  # shortcut 1 from here
        self.module_list.append(darknet_conv(in_ch=256, out_ch=512, ksize=3, stride=2))
        self.module_list.append(darknetblock(ch=512, nblocks=8))  # shortcut 2 from here
        self.module_list.append(darknet_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))
        self.module_list.append(darknetblock(ch=1024, nblocks=4))

        # YOLOv3
        self.module_list.append(darknetblock(ch=1024, nblocks=2, shortcut=False))
        self.module_list.append(darknet_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
        # 1st yolo branch
        self.module_list.append(darknet_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
        self.multi_scale_outputs=multi_scale_outputs
        if __C.VIS_PRETRAIN:
            parse_yolo_weights(self,__C.PRETRAIN_WEIGHT)
    def frozen(self,module):
        if getattr(module,'module',False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    def forward(self, x):
        outputs=[]
        for i,module in enumerate(self.module_list):
            x=module(x)
            if i in [6,8]:
                outputs.append(x)
        outputs.append(x)
        if self.multi_scale_outputs:
            return outputs
        else:
            return x

class VGG16(nn.Module):
    def __init__(self,__C, multi_scale_outputs=False):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.module_list.append(vgg_conv(in_ch=3, out_ch=64, ksize=3, stride=1))
        self.module_list.append(vgg_conv(in_ch=64, out_ch=64, ksize=3, stride=1))
        self.module_list.append(nn.MaxPool2d(2,2))
        self.module_list.append(vgg_conv(in_ch=64, out_ch=128, ksize=3, stride=1))
        self.module_list.append(vgg_conv(in_ch=128, out_ch=128, ksize=3, stride=1))
        self.module_list.append(nn.MaxPool2d(2,2))
        self.module_list.append(vgg_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
        self.module_list.append(vgg_conv(in_ch=256, out_ch=256, ksize=3, stride=1))
        self.module_list.append(vgg_conv(in_ch=256, out_ch=256, ksize=3, stride=1))
        self.module_list.append(nn.MaxPool2d(2,2)) # shortcut 1 from here
        self.module_list.append(vgg_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
        self.module_list.append(vgg_conv(in_ch=512, out_ch=512, ksize=3, stride=1))
        self.module_list.append(vgg_conv(in_ch=512, out_ch=512, ksize=3, stride=1))
        self.module_list.append(nn.MaxPool2d(2,2))  # shortcut 2 from here
        self.module_list.append(vgg_conv(in_ch=512, out_ch=512, ksize=3, stride=1))
        self.module_list.append(vgg_conv(in_ch=512, out_ch=512, ksize=3, stride=1))
        self.module_list.append(vgg_conv(in_ch=512, out_ch=512, ksize=3, stride=1))
        self.module_list.append(nn.MaxPool2d(2,2))
        # YOLOv3
        self.module_list.append(darknet_conv(in_ch=512, out_ch=512, ksize=1, stride=1))
        self.module_list.append(darknet_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
        self.module_list.append(darknet_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
        self.module_list.append(darknet_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
        self.module_list.append(darknet_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
        self.module_list.append(darknet_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
        self.multi_scale_outputs=multi_scale_outputs
        if __C.VIS_PRETRAIN:
            parse_yolo_weights(self,__C.PRETRAIN_WEIGHT)
    def forward(self, x):
        outputs=[]
        for i,module in enumerate(self.module_list):
            x=module(x)
            if i in [9,13]:
                outputs.append(x)
        outputs.append(x)
        if self.multi_scale_outputs:
            return outputs
        else:
            return x
class ResNet18(nn.Module):
    def __init__(self,__C, multi_scale_outputs=False):
        super().__init__()
        self.multi_scale_outputs=multi_scale_outputs
        self.resnet=resnet18(pretrained=__C.VIS_PRETRAIN)
    def forward(self, x):
        x=self.resnet.conv1(x)
        x=self.resnet.bn1(x)
        x=self.resnet.relu(x)
        x=self.resnet.maxpool(x)
        x=self.resnet.layer1(x)
        x0=self.resnet.layer2(x)
        x1=self.resnet.layer3(x0)
        x2=self.resnet.layer4(x1)
        if self.multi_scale_outputs:
            return [x0,x1,x2]
        return x2

class ResNet34(nn.Module):
    def __init__(self,__C, multi_scale_outputs=False):
        super().__init__()
        self.multi_scale_outputs=multi_scale_outputs
        self.resnet=resnet34(pretrained=__C.VIS_PRETRAIN)
    def forward(self, x):
        x=self.resnet.conv1(x)
        x=self.resnet.bn1(x)
        x=self.resnet.relu(x)
        x=self.resnet.maxpool(x)
        x=self.resnet.layer1(x)
        x0=self.resnet.layer2(x)
        x1=self.resnet.layer3(x0)
        x2=self.resnet.layer4(x1)
        if self.multi_scale_outputs:
            return [x0,x1,x2]
        return x2
class ResNet101(nn.Module):
    def __init__(self,__C, multi_scale_outputs=False):
        super().__init__()
        self.multi_scale_outputs=multi_scale_outputs
        self.resnet=resnet101(pretrained=__C.VIS_PRETRAIN)
    def forward(self, x):
        x=self.resnet.conv1(x)
        x=self.resnet.bn1(x)
        x=self.resnet.relu(x)
        x=self.resnet.maxpool(x)
        x=self.resnet.layer1(x)
        x0=self.resnet.layer2(x)
        x1=self.resnet.layer3(x0)
        x2=self.resnet.layer4(x1)
        if self.multi_scale_outputs:
            return [x0,x1,x2]
        return x2
backbone_dict={
    'vgg':VGG16,
    'darknet': DarkNet53,
    'resnet34': ResNet34,
    'resnet101':ResNet101,
    'resnet101d':ResNetV1c,
    'resnet18':ResNet18,
    'cspdarknet':CspDarkNet
}
def visual_encoder(__C):
    vis_enc=backbone_dict[__C.VIS_ENC](__C,multi_scale_outputs=True)
    return vis_enc

if __name__ == '__main__':
    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.USE_GLOVE = False
            self.WORD_EMBED_SIZE = 300
            self.HIDDEN_SIZE = 512
            self.N_SA = 0
            self.FLAT_GLIMPSES = 8
            self.DROPOUT_R = 0.1
            self.LANG_ENC = 'lstm'
            self.VIS_ENC = 'vgg'
            self.VIS_PRETRAIN = True
            self.PRETTRAIN_WEIGHT = './yolov3-vgg_450000.weights'
            self.ANCHORS = [[116, 90], [156, 198], [373, 326]]
            self.ANCH_MASK = [[0, 1, 2]]
            self.N_CLASSES = 0
            self.VIS_FREEZE = True
    cfg=Cfg()
    backbnone=VGG16(cfg)
    x=torch.zeros(2,3,416,416)
    y=backbnone(x)
    print(y.size())
