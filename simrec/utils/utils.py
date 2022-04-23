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


class EMA(object):
    '''
        apply expontential moving average to a model. This should have same function as the `tf.train.ExponentialMovingAverage` of tensorflow.
        usage:
            model = resnet()
            model.train()
            ema = EMA(model, 0.9999)
            ....
            for img, lb in dataloader:
                loss = ...
                loss.backward()
                optim.step()
                ema.update_params() # apply ema
            evaluate(model)  # evaluate with original model as usual
            ema.apply_shadow() # copy ema status to the model
            evaluate(model) # evaluate the model with ema paramters
            ema.restore() # resume the model parameters
        args:
            - model: the model that ema is applied
            - alpha: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
            - buffer_ema: whether the model buffers should be computed with ema method or just get kept
        methods:
            - update_params(): apply ema to the model, usually call after the optimizer.step() is called
            - apply_shadow(): copy the ema processed parameters to the model
            - restore(): restore the original model parameters, this would cancel the operation of apply_shadow()
    '''
    def __init__(self, model, alpha, buffer_ema=True):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = self.model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name]
                + (1 - decay) * state[name]
            )
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(
                    decay * self.shadow[name]
                    + (1 - decay) * state[name]
                )
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }
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

def setup_unique_version(cfg):
    while True:
        version = random.randint(0, 99999)
        if not (os.path.exists(os.path.join(cfg.train.log_path ,str(version)))):
            cfg.train.version = str(version)
            break

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port



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

def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

def get_lr_scheduler(__C,optimizer):
    if __C.SCHEDULER == 'step':
        t,T=__C.WARMUP,__C.EPOCHS
        def lr_func(epoch):
            coef = 1.
            if epoch<=t:
                coef=float(epoch)/float(t+1)
            else:
                for i,deps in enumerate(__C.DECAY_EPOCHS):
                    if epoch>=deps:
                        coef=__C.LR_DECAY_R**(i+1)
            return coef
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch:lr_func(epoch))
    elif __C.SCHEDULER == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=__C.EPOCHS)
    else:
        t,T=__C.WARMUP,__C.EPOCHS
        n_t=0.5
        lr_func = lambda epoch: (0.9 * epoch / t + __C.LR) if epoch < t else __C.LR if n_t * (
                    1 + math.cos(math.pi * (epoch - t) / (T - t))) < __C.LR else n_t * (
                    1 + math.cos(math.pi * (epoch - t) / (T - t)))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler

def get_lr_scheduler(__C,optimizer,n_iter_per_epoch):
    num_steps = int(__C.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(__C.WARMUP * n_iter_per_epoch)
    if __C.SCHEDULER == 'step':
        #default step lr
        t,T=__C.WARMUP*n_iter_per_epoch,__C.EPOCHS*n_iter_per_epoch
        def lr_func(step):
            coef = 1.
            if step<=t:
                coef=float(step)/float(t+1)
            else:
                for i,deps in enumerate(__C.DECAY_EPOCHS):
                    if step>=deps*n_iter_per_epoch:
                        coef=__C.LR_DECAY_R**(i+1)
            return coef
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif __C.SCHEDULER == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=__C.EPOCHS*n_iter_per_epoch)
    else:
        t, T = __C.WARMUP * n_iter_per_epoch, __C.EPOCHS * n_iter_per_epoch
        n_t=0.5
        warm_step_lr=(__C.LR - __C.WARMUP_LR) / t
        lr_func = lambda step: ( step*warm_step_lr + __C.WARMUP_LR)/__C.LR if step < t \
            else (__C.MIN_LR + n_t * (__C.LR - __C.MIN_LR) * (1 + math.cos(math.pi * (step - t) / (T - t))))/__C.LR

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler