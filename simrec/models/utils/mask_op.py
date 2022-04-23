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

import cv2
import numpy as np


def mask_iou(mask1, mask2):
    """
    :param mask1:  l
    :param mask2:  l
    :return: iou
    """
    mask1 =mask1.reshape([-1])
    mask2=mask2.reshape([-1])
    t = np.array(mask1 > 0.5)
    p = mask2 > 0.
    intersection = np.logical_and(t, p)
    union = np.logical_or(t, p)
    iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)

    ap = dict()
    thresholds = np.arange(0.5, 1, 0.05)
    s = []
    for thresh in thresholds:
        ap[thresh] = float(iou > thresh)
    return iou,ap


def mask_processing(mask,info_img):
    # print(info_img)
    h, w, nh, nw, dx, dy,_=info_img
    # print(info_img)
    # print(mask)
    mask=mask[dy:dy + nh, dx:dx + nw,None]
    mask=cv2.resize(mask,(int(w),int(h)))
    return mask