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
from torch.utils.data import DistributedSampler, SequentialSampler
from torch.utils.data import DataLoader


def build_train_loader(cfg, dataset: torch.utils.data.Dataset, shuffle=True, drop_last=False):
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    assert cfg.train.batch_size % num_tasks == 0
    assert dist.is_initialized()

    train_micro_batch_size = cfg.train.batch_size // num_tasks

    train_sampler = DistributedSampler(
                                dataset,
                                num_replicas=num_tasks,
                                shuffle=shuffle,
                                rank=global_rank,
                                )
    data_loader = DataLoader(
                            dataset,
                            batch_size=train_micro_batch_size,
                            sampler=train_sampler,
                            num_workers=cfg.train.data.num_workers,
                            pin_memory=cfg.train.data.pin_memory,
                            drop_last=drop_last
                        )
    return data_loader


def build_test_loader(cfg, dataset: torch.utils.data.Dataset, shuffle=False, drop_last=False):
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    assert cfg.train.evaluation.eval_batch_size % num_tasks == 0
    assert dist.is_initialized()

    eval_micro_batch_size = cfg.train.evaluation.eval_batch_size // num_tasks

    if cfg.train.evaluation.sequential:
        eval_micro_batch_size = cfg.train.evaluation.eval_batch_size
        eval_sampler = SequentialSampler(dataset)
    else:
        eval_sampler = DistributedSampler(
                                    dataset,
                                    num_replicas=num_tasks,
                                    shuffle=shuffle,
                                    rank=global_rank,
                                    )
    
    data_loader = DataLoader(
                            dataset,
                            batch_size=eval_micro_batch_size,
                            sampler=eval_sampler,
                            num_workers=cfg.train.data.num_workers,
                            pin_memory=cfg.train.data.pin_memory,
                            drop_last=drop_last
                        )
    return data_loader
