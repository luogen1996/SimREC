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

import torch


def normed2original(image,mean=None,std=None,transpose=True):
    """
    :param image: 3,h,w
    :param mean: 3
    :param std: 3
    :return:
    """
    if std is not None:
        std=torch.from_numpy(np.array(std)).to(image.device).float()
        image=image*std.unsqueeze(-1).unsqueeze(-1)
    if mean is not None:
        mean=torch.from_numpy(np.array(mean)).to(image.device).float()
        image=image+mean.unsqueeze(-1).unsqueeze(-1)
    if transpose:
        image=image.permute(1,2,0)
    return image.cpu().numpy()


def draw_visualization(image,sent,pred_box,gt_box,draw_text=True,savepath=None):
    # image=(image*255).astype(np.uint8)
    image=np.ascontiguousarray(image)
    left, top, right, bottom,_ = (pred_box).astype('int32')
    gt_left, gt_top, gt_right, gt_bottom = (gt_box).astype('int32')
    colors=[(255,0,0),(0,255,0),(0,191,255)]

    cv2.rectangle(image, (left, top ), (right , bottom ), colors[0], 2)
    cv2.rectangle(image, (gt_left, gt_top), (gt_right, gt_bottom), colors[1], 2)
    # cv2.imwrite(savepath+str(k)+'.jpg',img)


    if draw_text:
        cv2.putText(image,
                    '{:%.2f}' % pred_box[-1],
                    (left, max(top - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[0], 2)
        cv2.putText(image,
                    'ground_truth',
                    (gt_left, max(gt_top - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[1], 2)
        cv2.putText(image,
                    str(sent),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[2], 2)
    return image