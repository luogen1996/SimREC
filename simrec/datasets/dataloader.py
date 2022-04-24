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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def build_loader(cfg, dataset: torch.utils.data.Dataset, rank: int, shuffle=True, drop_last=False):
    assert cfg.train.batch_size % len(cfg.train.gpus) == 0
    assert cfg.train.num_workers % len(cfg.train.gpus) == 0
    assert dist.is_initialized()
    dist_sampler = DistributedSampler(
                                dataset,
                                num_replicas=dist.get_world_size(),
                                shuffle=shuffle,
                                rank=rank,
                                )
    data_loader = DataLoader(
                            dataset,
                            batch_size=cfg.train.batch_size // len(cfg.train.gpus),
                            sampler=dist_sampler,
                            num_workers=cfg.train.num_workers //len(cfg.train.gpus),
                            pin_memory=True,
                            drop_last=drop_last
                        )
    return data_loader

