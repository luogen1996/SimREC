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

import numpy as np

def label2yolobox(labels, info_img, maxsize, lrflip=False):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    h, w, nh, nw, dx, dy,_ = info_img
    x1 = labels[:, 0] / w
    y1 = labels[:, 1] / h
    x2 = (labels[:, 0] + labels[:, 2]) / w
    y2 = (labels[:, 1] + labels[:, 3]) / h
    labels[:, 0] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 1] = (((y1 + y2) / 2) * nh + dy) / maxsize
    labels[:, 2] *= nw / w / maxsize
    labels[:, 3] *= nh / h / maxsize
    labels[:,:4]=np.clip(labels[:,:4],0.,0.99)
    if lrflip:
        labels[:, 0] = 1 - labels[:, 0]
    return labels


def yolobox2label(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    # h, w, nh, nw, dx, dy,_ = info_img
    # y1, x1, y2, x2 = box
    # box_h = ((y2 - y1) / nh) * h
    # box_w = ((x2 - x1) / nw) * w
    # y1 = ((y1 - dy) / nh) * h
    # x1 = ((x1 - dx) / nw) * w
    # label = [y1, x1, y1 + box_h, x1 + box_w]
    h, w, nh, nw, dx, dy,_ = info_img
    x1, y1, x2, y2 = box[:4]
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    label = [x1, y1,x1 + box_w, y1 + box_h]
    return np.concatenate([np.array(label),box[4:]])