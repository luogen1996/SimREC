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

import os
import numpy as np
import warnings
import random
from typing import Optional

import torch as th
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from utils.logging import AverageMeter


def seed_everything(SEED: Optional[int]):
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

        th.manual_seed(SEED)
        th.cuda.manual_seed(SEED)
        if th.cuda.device_count() > 1:
            th.cuda.manual_seed_all(SEED)

        cudnn.benchmark = False
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def setup_gpu_env():
    assert 'CUDA_DEVICE_ORDER' in os.environ.keys() and 'CUDA_VISIBLE_DEVICES' in os.environ.keys(), \
        "set CUDA_DEVICE_ORDER and CUDE_VISIBLE_DEVICES environment variable before executing"
    GPUs = os.environ['CUDA_VISIBLE_DEVICES']

    _GPUs = [int(idx) for idx in GPUs if idx.isdigit()]
    _USEs = [idx for idx in range(len(_GPUs))]

    return _USEs


def main_process(__C,rank):
    return not __C.MULTIPROCESSING_DISTRIBUTED or (__C.MULTIPROCESSING_DISTRIBUTED and rank == 0)


def setup_distributed(__C,rank: int, backend: str = 'NCCL'):
    if not dist.is_available():
        raise ModuleNotFoundError('torch.distributed package not found')

    if __C.WORLD_SIZE > len(__C.GPU):
        assert '127.0.0.1' not in __C.DIST_URL, "DIST_URL is illegal with multi nodes distributed training"

    dist.init_process_group(dist.Backend(backend), rank=rank, world_size=__C.WORLD_SIZE, init_method=__C.DIST_URL)

    if not dist.is_initialized():
        raise ValueError('init_process_group failed')


def cleanup_distributed():
    dist.destroy_process_group()


def reduce_meters(meters, rank, __C):
    """Sync and flush meters."""
    assert isinstance(meters, dict), "collect AverageMeters into a dict"
    for name in sorted(meters.keys()):
        meter = meters[name]
        if not isinstance(meter, AverageMeter):
            raise TypeError("meter should be AverageMeter type")
        if not __C.MULTIPROCESSING_DISTRIBUTED:  # single gpu
            meter.update_reduce(meter.avg)
        else:
            avg = th.tensor(meter.avg).unsqueeze(0).to(rank)
            avg_reduce = [th.ones_like(avg) for _ in range(dist.get_world_size())]
            # print("rank {} gathering {} meter".format(rank, name))
            # print("rank {}, avg {}, avg_reduce {}".format(rank, avg, avg_reduce))
            dist.all_gather(avg_reduce, avg)
            if main_process(__C,rank):
                value = th.mean(th.cat(avg_reduce)).item()
                meter.update_reduce(value)