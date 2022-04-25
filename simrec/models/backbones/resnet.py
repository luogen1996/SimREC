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

from torchvision.models.resnet import resnet34, resnet101, resnet18

class ResNet18(nn.Module):
    def __init__(
        self,
        pretrained=False, 
        multi_scale_outputs=False,
        freeze_backbone=True,
    ):
        super().__init__()
        self.multi_scale_outputs=multi_scale_outputs
        self.resnet=resnet18(pretrained=pretrained)

        if freeze_backbone:
            self.frozen()

    def frozen(self):
        for module in self.modules():
            for name, param in module.named_parameters():
                param.requires_grad=False


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
    def __init__(
        self,
        pretrained=False, 
        multi_scale_outputs=False,
        freeze_backbone=True,
    ):
        super().__init__()
        self.multi_scale_outputs=multi_scale_outputs
        self.resnet=resnet34(pretrained=pretrained)
    
        if freeze_backbone:
            self.frozen()

    def frozen(self):
        for module in self.modules():
            for name, param in module.named_parameters():
                param.requires_grad=False

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
    def __init__(
        self,
        pretrained=False, 
        multi_scale_outputs=False,
        freeze_backbone=True,
    ):
        super().__init__()
        self.multi_scale_outputs=multi_scale_outputs
        self.resnet=resnet101(pretrained=pretrained)

        if freeze_backbone:
            self.frozen()

    def frozen(self):
        for module in self.modules():
            for name, param in module.named_parameters():
                param.requires_grad=False

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
