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

import torch.nn as nn

from simrec.layers.blocks import vgg_conv, darknet_conv
from simrec.utils.parse_darknet_weights import parse_yolo_weights

class VGG16(nn.Module):
    def __init__(
        self,
        pretrained_weight_path=None,
        pretrained=False, 
        multi_scale_outputs=False,
        freeze_backbone=True,
    ):
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
        
        if pretrained:
            parse_yolo_weights(pretrained_weight_path)

        if freeze_backbone:
            self.frozen(self.module_list[:-2])
        
    def frozen(self, module):
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
            if i in [9,13]:
                outputs.append(x)
        outputs.append(x)
        if self.multi_scale_outputs:
            return outputs
        else:
            return x