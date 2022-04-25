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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.weight * (x - mean) / (std + self.eps) + self.bias

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class AttFlat(nn.Module):
    def __init__(self, hidden_size, flat_glimpses, dropout_rate):
        super(AttFlat, self).__init__()
        self.hidden_size = hidden_size
        self.flat_glimpses = flat_glimpses
        self.dropout_rate = dropout_rate

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=hidden_size//2,
            out_size=flat_glimpses,
            dropout_r=dropout_rate,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size ,
            hidden_size
        )

    def forward(self, x, x_mask=None):
        b, l, c = x.size()
        att = self.mlp(x).view(b,l,-1)
        x=x.view(b, l, self.flat_glimpses, -1)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e4
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x[:,:,i,:], dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super(MHAtt, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_merge = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.hidden_size / self.num_heads)
        ).transpose(1, 2)
        
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.hidden_size / self.num_heads)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.num_heads,
            int(self.hidden_size / self.num_heads)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        # print(scores.size(),mask.size())
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ffn_size,
            out_size=hidden_size,
            dropout_r=dropout_rate,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_size, dropout_rate):
        super(SA, self).__init__()

        self.mhatt = MHAtt(hidden_size, num_heads, dropout_rate)
        self.ffn = FFN(hidden_size, ffn_size, dropout_rate)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(hidden_size)
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, y, y_mask,pos=None):
        q=k= self.with_pos_embed(y, pos)
        y = self.norm1(y + self.dropout1(
            self.mhatt(v=y, k=k, q=q, mask=y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_size, dropout_rate):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(hidden_size, num_heads, dropout_rate)
        self.mhatt2 = MHAtt(hidden_size, num_heads, dropout_rate)
        self.ffn = FFN(hidden_size, ffn_size, dropout_rate)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(hidden_size)

        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = LayerNorm(hidden_size)
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, x, y, x_mask, y_mask,q_pos=None,k_pos=None):
        q=k= self.with_pos_embed(x, q_pos)
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=k, q=q, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=self.with_pos_embed(y, k_pos), q=self.with_pos_embed(x, q_pos), mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

