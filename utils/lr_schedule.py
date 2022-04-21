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

class LR_Scheduler:

    def __init__(self, schedule_func, tensorboard, init_epoch=0, verbose=0):
        super(LR_Scheduler, self).__init__()
        self.schedule_func = schedule_func
        self.verbose = verbose
        self.epoch = init_epoch
        self.lr = 0.

    def __call__(self, epoch, optimizer):
        self.epoch += 1
        self.lr = self.schedule_func(self.epoch)
        for param_group in optimizer.param_groups():
            param_group['lr'] = self.lr
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %.4f' % (self.epoch, self.lr))
