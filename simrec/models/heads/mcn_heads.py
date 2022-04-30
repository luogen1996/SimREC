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

import torch
import torch.nn as nn
import torch.nn.functional as F

from simrec.layers.aspp import aspp_decoder
from ..utils.box_op import bboxes_iou


class MCNhead(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(
        self, 
        hidden_size=512, 
        anchors=[[137, 256], [248, 272], [386, 271]], 
        arch_mask=[[0, 1, 2]], 
        layer_no=0, 
        in_ch=512, 
        n_classes=0, 
        ignore_thre=0.5
    ):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(MCNhead, self).__init__()
        self.anchors = anchors
        self.anch_mask = arch_mask[layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = n_classes
        self.ignore_thre = ignore_thre
        self.l2_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.stride = 32 # strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.d_proj = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        self.s_proj = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

        self.dconv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_classes + 5),
                              kernel_size=1, stride=1, padding=0)
        self.sconv=nn.Sequential(aspp_decoder(in_ch, hidden_size//2, 1),
                                 nn.UpsamplingBilinear2d(scale_factor=8)
                                )


    def nls(self,pred_seg,pred_box,weight_score=None,lamb_au=-1.,lamb_bu=2,lamb_ad=1.,lamb_bd=0):
        if weight_score is not None:
            #asnls
            mask = torch.ones_like(pred_seg)*weight_score.unsqueeze(1).unsqueeze(1)*lamb_ad+lamb_bd
            pred_box=pred_box[:,:4].long()
            for i in range(pred_seg.size()[0]):
                mask[i,pred_box[i,1]:pred_box[i,3] + 1, pred_box[i,0]:pred_box[i,2] + 1, ...]=weight_score[i].item()*lamb_au+lamb_bu
        else:
            #hard-nls
            mask=torch.zeros_like(pred_seg)
            pred_box = pred_box[:, :4].long()
            for i in range(pred_seg.size()[0]):
                mask[i,pred_box[i][1]:pred_box[i][3]+1,pred_box[i][0]:pred_box[i][2]+1]=1.
        return pred_seg*mask
    
    def co_energe(self,x_map,y_map,x_attn,y_attn,eps=1e-6):
        """
        :param x_map:  h*w
        :param y_map: h*w
        :param x_attn: B,c,h,w
        :param y_attn: B,c,h,w
        :return:
        """
        b,c,h,w=x_attn.size()
        x_map=F.softmax(self.s_proj(x_attn).view(b,-1),-1)
        y_map=F.softmax(self.d_proj(y_attn).view(b,-1),-1)
        x_attn=F.normalize(x_attn,dim=1).view(b,c,-1)
        y_attn=F.normalize(y_attn,dim=1).view(b,c,-1)
        cosin_sim=torch.bmm(x_attn.transpose(1,2),y_attn) #b,h_x*w_x,h_y*w_y
        cosin_sim=(cosin_sim+1.)/2.
        co_en=torch.einsum('blk,bl,bk->b',[cosin_sim,x_map,y_map])
        return -torch.log(co_en+eps)
    
    def forward(self, xin,yin, x_label=None,y_label=None,x_map=None,y_map=None,x_attn=None,y_attn=None):
        output = self.dconv(xin)
        mask=self.sconv(yin)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor
        devices=xin.device

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            output[..., np.r_[:2, 4:n_ch]])

        # calculate pred - xywh obj cls

        x_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4])).to(devices)
        y_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4])).to(devices)

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4])).to(devices)
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4])).to(devices)

        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        if not self.training:
            pred[..., :4] *= self.stride
            pred=pred.view(batchsize,-1,n_ch)
            #xc,yc,,w,h->xmin,ymin,xmax,ymax
            pred[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
            pred[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
            pred[:, :, 2] = pred[:, :, 0] + pred[:, :, 2]
            pred[:, :, 3] = pred[:, :, 1] + pred[:, :, 3]
            score=pred[:,:,4].sigmoid()
            max_score,ind=torch.max(score,-1)
            ind=ind.unsqueeze(1).unsqueeze(1).repeat(1,1,n_ch)
            pred=torch.gather(pred,1,ind)

            mask=(mask.squeeze(1).sigmoid()>0.35).float()
            return pred.view(batchsize,-1),mask

        pred = pred[..., :4].data

        # target assignment

        tgt_mask = torch.zeros(batchsize, self.n_anchors,
                               fsize, fsize, 4 + self.n_classes).type(dtype).to(devices)
        obj_mask = torch.ones(batchsize, self.n_anchors,
                              fsize, fsize).type(dtype).to(devices)
        tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 2).type(dtype).to(devices)

        target = torch.zeros(batchsize, self.n_anchors,
                             fsize, fsize, n_ch).type(dtype).to(devices)

        x_label = x_label.cpu().data
        nlabel = (x_label.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = x_label[:, :, 0] * fsize
        truth_y_all = x_label[:, :, 1] * fsize
        truth_w_all = x_label[:, :, 2] * fsize
        truth_h_all = x_label[:, :, 3] * fsize
        truth_i_all = truth_x_all.to(torch.int16).numpy()
        truth_j_all = truth_y_all.to(torch.int16).numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = dtype(np.zeros((n, 4))).to(devices)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_mask[0]) | (
                best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(
                pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = 1 - pred_best_iou.float()

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - \
                        truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - \
                        truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    # target[b, a, j, i, 5 + x_label[b, ti,
                    #                               0].to(torch.int16).numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

            # loss calculation

        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale
        # print(output.size(),target.size())

        bceloss = nn.BCELoss(weight=tgt_scale*tgt_scale,reduction='none')  # weighted BCEloss
        loss_xy = bceloss(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])


        loss_seg=nn.BCEWithLogitsLoss(reduction='sum')(mask,y_label)/640./batchsize
        loss_det = loss_xy.sum() + loss_wh.sum() + loss_obj.sum()
        loss_det/=float(batchsize)
        loss_cem = self.co_energe(x_map, y_map, x_attn, y_attn)
        loss = loss_det.sum() + loss_seg.sum() + loss_cem.sum()
        return loss, loss_det, loss_seg
