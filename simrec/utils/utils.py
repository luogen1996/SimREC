"""Miscellaneous utility functions."""

from functools import reduce
import torch
from PIL import Image
import numpy as np
import re
import cv2
import os
import random
import torch.optim.lr_scheduler as lr_scheduler
import math
import torch.nn as nn

# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Guodong Zhang
# --------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
#%matplotlib inline
import matplotlib.cm as cm
import numpy as np


def getLayers(model):
    """
    get each layer's name and its module
    :param model:
    :return: each layer's name and its module
    """
    layers = []

    def unfoldLayer(model):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """

        # get all layers of the model
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            # if current layer contains sublayers, add current layer name on its sublayers
            if sublayer_num == 0 and len(list(module.parameters()))>0:
                layers.append(module)
            # if current layer contains sublayers, unfold them
            elif isinstance(module, nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)
    return layers


def get_module_ops(obj):
    ret=[]
    for m in obj:
        if hasattr(obj,'children'):
            ret+=get_module_ops(m)
        else:
            ret.append(m)
    return  ret

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []
    total=getLayers(net)
    # for (i,j) in net.named_parameters():
    #     print(i)
    for m in total:
        # print(m)
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                no_decay.append(m.bias)
        else:
            no_decay+=list(m.parameters())
            # print(m)
            # if hasattr(m, 'weight'):
            #     no_decay.append(m.weight)
            # elif hasattr(m, 'bias'):
            #     no_decay.append(m.bias)
            # else:
            #     no_decay.append(m)

    # print(len(list(net.parameters())),len(total),len(decay) , len(no_decay))
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=filter(lambda p: p.requires_grad, decay)), dict(params=filter(lambda p: p.requires_grad, no_decay), weight_decay=0)]






def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)