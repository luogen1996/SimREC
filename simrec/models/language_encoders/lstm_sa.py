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

from simrec.layers.sa_layer import SA, AttFlat


def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


class LSTM_SA(nn.Module):
    """
    Building LSTM - SelfAttention layer

    Args:
        depth (int): the depth of self-attention layers
        hidden_size (int): the size of the hidden states
        num_heads (int): number of attention heads
        ffn_size (int): size for the feed-forward nets
        freeze_embedding (bool): freeze the weights of embedding layer or not
    """

    def __init__(
        self, 
        depth,
        hidden_size,
        num_heads,
        ffn_size,
        flat_glimpses,
        dropout_rate,
        word_embed_size,
        pretrained_emb, 
        token_size,
        freeze_embedding=True,
        use_glove=True,
    ):
        super(LSTM_SA, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=word_embed_size
        )

        # Loading the GloVe embedding weights
        if use_glove:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.GRU(
            input_size=word_embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=False
        )

        self.sa_list = nn.ModuleList([SA(hidden_size, num_heads, ffn_size, dropout_rate) for _ in range(depth)])
        self.att_flat=AttFlat(hidden_size, flat_glimpses, dropout_rate)
        
        if freeze_embedding:
            self.frozen(self.embedding)
    
    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    
    def forward(self, ques_ix):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)


        for sa in self.sa_list:
            lang_feat = sa(lang_feat, lang_feat_mask)

        flat_lang_feat = self.att_flat(lang_feat, lang_feat_mask)
        return  {
            'flat_lang_feat':flat_lang_feat,
            'lang_feat':lang_feat,
            'lang_feat_mask':lang_feat_mask
        }
