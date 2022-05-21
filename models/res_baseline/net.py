import torch
import torch.nn as nn
from models.res_baseline.head   import REShead
from models.language_encoder import language_encoder
from models.visual_encoder import visual_encoder
from layers.fusion_layer import SimpleFusion,MultiScaleFusion,GaranAttention

import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.visual_encoder=visual_encoder(__C)
        self.lang_encoder=language_encoder(__C,pretrained_emb,token_size)
        self.fusion_manner=SimpleFusion(q_planes=512)
        self.multi_scale_manner=MultiScaleFusion(v_planes=[256,512,1024])
        self.seg_garran=GaranAttention(512,512)
        self.det_garan=GaranAttention(512,512)

        self.head=REShead(__C,0,__C.HIDDEN_SIZE)
        total = sum([param.nelement() for param in self.fusion_manner.parameters()])
        print('  + Number of fusion_manner params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in self.multi_scale_manner.parameters()])
        print('  + Number of multi_scale_manner params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in self.det_garan.parameters()])
        print('  + Number of det_garan params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        if __C.VIS_FREEZE:
            if __C.VIS_ENC=='vgg' or __C.VIS_ENC=='darknet':
                self.frozen(self.visual_encoder.module_list[:-2])
            else:
                self.frozen(self.visual_encoder)
    def frozen(self,module):
        if getattr(module,'module',False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    def forward(self, images,y, det_label=None,seg_label=None):
        x=self.visual_encoder(images)
        y=self.lang_encoder(y)
        x[-1]=self.fusion_manner(x[-1],y['flat_lang_feat'])
        bot_feats,_,top_feats=self.multi_scale_manner(x)
        bot_feats,seg_map,seg_attn=self.seg_garran(y['flat_lang_feat'],bot_feats)
        if self.training:
            loss,loss_det,loss_seg=self.head(top_feats,bot_feats,det_label,seg_label)
            return loss,loss_det,loss_seg
        else:
            box, mask=self.head(top_feats,bot_feats)
            return box,mask


if __name__ == '__main__':
    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.USE_GLOVE = False
            self.WORD_EMBED_SIZE = 300
            self.HIDDEN_SIZE = 512
            self.N_SA = 0
            self.FLAT_GLIMPSES = 8
            self.DROPOUT_R = 0.1
            self.LANG_ENC = 'lstm'
            self.VIS_ENC = 'darknet'
            self.VIS_PRETRAIN = True
            self.PRETTRAIN_WEIGHT = './darknet.weights'
            self.ANCHORS = [[116, 90], [156, 198], [373, 326]]
            self.ANCH_MASK = [[0, 1, 2]]
            self.N_CLASSES = 0
            self.VIS_FREEZE = True
    cfg=Cfg()
    model=Net(cfg,torch.zeros(1),100)
    # model.train()
    img=torch.zeros(2,3,224,224)
    lang=torch.randint(10,(2,14))
    seg, det=model(img,lang)
    print(seg.size(),det.size())