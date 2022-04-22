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

from simrec.models.heads.rec_heads import REChead
from simrec.models.language_encoders.build import build_language_encoder
from simrec.models.backbones.build import build_visual_encoder
from simrec.layers.fusion_layer import MultiScaleFusion,SimpleFusion,GaranAttention

torch.backends.cudnn.enabled=False

class SimREC(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(SimREC, self).__init__()
        self.visual_encoder=build_visual_encoder(__C)
        self.lang_encoder=build_language_encoder(__C,pretrained_emb,token_size)
        self.multi_scale_manner = MultiScaleFusion(v_planes=(512, 512, __C.HIDDEN_SIZE), scaled=True)
        self.fusion_manner=nn.ModuleList(
            [
                SimpleFusion(v_planes=256, out_planes=512, q_planes=512),
                SimpleFusion(v_planes=512, out_planes=512, q_planes=512),
                SimpleFusion(v_planes=1024, out_planes=512, q_planes=512)
            ]
        )
        self.attention_manner=GaranAttention(512,512)
        self.head=REChead(__C)
        total = sum([param.nelement() for param in self.lang_encoder.parameters()])
        print('  + Number of lang enc params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        if __C.VIS_FREEZE:
            if __C.VIS_ENC=='vgg' or __C.VIS_ENC=='darknet':
                self.frozen(self.visual_encoder.module_list[:-2])
            elif __C.VIS_ENC=='cspdarknet':
                self.frozen(self.visual_encoder.model[:-2])
            else:
                self.frozen(self.visual_encoder)
    
    def frozen(self,module):
        if getattr(module,'module',False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    
    def forward(self,x,y, det_label=None,seg_label=None):

        x=self.visual_encoder(x)
        
        y=self.lang_encoder(y)
        
        for i in range(len(self.fusion_manner)):
            x[i]=self.fusion_manner[i](x[i],y['flat_lang_feat'])
        
        x=self.multi_scale_manner(x)
        
        top_feats,_,_=self.attention_manner(y['flat_lang_feat'],x[-1])
        
        bot_feats=x[0]
        
        if self.training:
            loss,loss_det,loss_seg=self.head(top_feats,bot_feats,det_label,seg_label)
            return loss,loss_det,loss_seg
        
        else:
            box, mask=self.head(top_feats,bot_feats)
            return box,mask
