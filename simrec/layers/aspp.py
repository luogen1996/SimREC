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

import torch
import torch.nn as nn

from .blocks import darknet_conv


class aspp_decoder(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Layer

    Args:
        planes (int): input channels
        hidden_planes (int): middle channels
        out_planes (int): output channels
    """
    def __init__(self, planes,hidden_planes,out_planes):
        super().__init__()
        self.conv0 = darknet_conv(planes, hidden_planes, ksize=1, stride=1)
        self.conv1 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=6)
        self.conv2 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=12)
        self.conv3 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=18)
        self.conv4 = darknet_conv(planes, hidden_planes, ksize=1, stride=1)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.out_proj= nn.Conv2d(hidden_planes*5, out_planes, 1)
    def forward(self, x):
        _, _, h, w = x.size()
        b0 = self.conv0(x)
        b1 = self.conv1(x)
        b2 = self.conv2(x)
        b3 = self.conv3(x)
        b4 = self.conv4(self.pool(x)).repeat(1,1,h,w)
        x=torch.cat([b0,b1,b2,b3,b4],1)
        x=self.out_proj(x)
        return x