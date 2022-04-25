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

from simrec.models.heads.mcn_heads import MCNhead
from simrec.models.backbones.build import build_visual_encoder
from simrec.models.language_encoders.build import build_language_encoder
from simrec.layers.fusion_layer import SimpleFusion, MultiScaleFusion, GaranAttention


class MCN(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(MCN, self).__init__()
        self.visual_encoder=build_visual_encoder(__C)
        self.lang_encoder=build_language_encoder(__C, pretrained_emb,token_size)
        self.fusion_manner=SimpleFusion()
        self.multi_scale_manner=MultiScaleFusion(v_planes=[256,512,1024])
        self.seg_garran=GaranAttention(512,512)
        self.det_garan=GaranAttention(512,512)

        self.head=MCNhead(__C,0,__C.HIDDEN_SIZE)
        total = sum([param.nelement() for param in self.fusion_manner.parameters()])
        print('  + Number of fusion_manner params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in self.multi_scale_manner.parameters()])
        print('  + Number of multi_scale_manner params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in self.det_garan.parameters()])
        print('  + Number of det_garan params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        if __C.VIS_FREEZE:
            if __C.VIS_ENC=='vgg' or __C.VIS_ENC=='darknet':
                self.frozen(self.visual_encoder.module_list[:-2])
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
    
    def forward(self, x,y, det_label=None,seg_label=None):
        x=self.visual_encoder(x)
        y=self.lang_encoder(y)
        x[-1]=self.fusion_manner(x[-1],y['flat_lang_feat'])
        bot_feats,top_feats=self.multi_scale_manner(x)
        bot_feats,seg_map,seg_attn=self.seg_garran(y['flat_lang_feat'],bot_feats)
        top_feats,det_map,det_attn=self.det_garan(y['flat_lang_feat'],top_feats)
        if self.training:
            loss,loss_det,loss_seg=self.head(top_feats,bot_feats,det_label,seg_label,det_map,seg_map,det_attn,seg_attn)
            return loss,loss_det,loss_seg
        else:
            box, mask=self.head(top_feats,bot_feats)
            return box,mask
