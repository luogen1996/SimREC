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


def smooth_L1(y_true, y_pred,sigma=3.0):
    sigma_squared = sigma ** 2

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = y_true - y_pred
    regression_diff = torch.abs(regression_diff)
    regression_loss = torch.where(
        regression_diff<(1.0 / sigma_squared),
        0.5 * sigma_squared * torch.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )
    return regression_loss.sum()