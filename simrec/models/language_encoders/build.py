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

from .lstm_sa import LSTM_SA

backbone_dict={
    'lstm':LSTM_SA,
}

def language_encoder(__C, pretrained_emb, token_size):
    lang_enc=backbone_dict[__C.LANG_ENC](__C, pretrained_emb, token_size)
    return lang_enc