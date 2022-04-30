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

from .lr_scheduler import StepLR, WarmupCosineLR

def build_lr_scheduler(cfg, optimizer, n_iter_per_epoch):
    """Build learning rate scheduler."""
    scheduler_name = cfg.train.scheduler.name.lower()
    
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = WarmupCosineLR(
            optimizer=optimizer,
            warmup_epochs=cfg.train.warmup_epochs,
            epochs=cfg.train.epochs,
            warmup_lr=cfg.train.warmup_lr,
            base_lr=cfg.train.base_lr,
            min_lr=cfg.train.min_lr,
            n_iter_per_epoch=n_iter_per_epoch
        )
    elif scheduler_name == "step":
        scheduler = StepLR(
            optimizer=optimizer,
            warmup_epochs=cfg.train.warmup_epochs,
            epochs=cfg.train.epochs,
            decay_epochs=cfg.train.scheduler.decay_epochs,
            lr_decay_rate=cfg.train.scheduler.lr_decay_rate,
            n_iter_per_epoch=n_iter_per_epoch,
        )
    
    return scheduler
