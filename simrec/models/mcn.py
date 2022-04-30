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

class MCN(nn.Module):
    def __init__(
        self, 
        visual_backbone: nn.Module,
        language_encoder: nn.Module,
        multi_scale_manner: nn.Module,
        fusion_manner: nn.Module,
        det_attention: nn.Module,
        seg_attention: nn.Module,
        head: nn.Module,
    ):
        super(MCN, self).__init__()
        self.visual_encoder = visual_backbone
        self.lang_encoder=language_encoder
        self.fusion_manner=fusion_manner
        self.multi_scale_manner=multi_scale_manner
        self.det_garan=det_attention
        self.seg_garan=seg_attention
        self.head=head
    
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

        x[-1]=self.fusion_manner(x[-1] ,y['flat_lang_feat'])
        bot_feats, _, top_feats=self.multi_scale_manner(x)
        bot_feats,seg_map,seg_attn=self.seg_garan(y['flat_lang_feat'],bot_feats)
        top_feats,det_map,det_attn=self.det_garan(y['flat_lang_feat'],top_feats)
        
        if self.training:
            loss,loss_det,loss_seg=self.head(top_feats,bot_feats,det_label,seg_label,det_map,seg_map,det_attn,seg_attn)
            return loss,loss_det,loss_seg
        else:
            box, mask=self.head(top_feats,bot_feats)
            return box,mask
