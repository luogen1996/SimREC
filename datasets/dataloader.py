# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import numpy as np
import  json, re, en_vectors_web_lg,random
import torch
import torch.utils.data as Data
import os
from utils.utils import label2yolobox
import cv2
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import transforms
import albumentations as A
from datasets.randaug import RandAugment

from torch.utils.data import DataLoader

class RefCOCODataSet(Data.Dataset):
    def __init__(self, __C,split):
        super(RefCOCODataSet, self).__init__()
        self.__C = __C
        self.split=split
        assert  __C.DATASET in ['refcoco', 'refcoco+', 'refcocog','referit','vg','merge']
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET], 'r'))
        total_refs_list=[]
        if __C.DATASET in ['vg','merge']:
            total_refs_list = json.load(open(__C.ANN_PATH['merge'], 'r'))+json.load(open(__C.ANN_PATH['refcoco+'], 'r'))+json.load(open(__C.ANN_PATH['refcocog'], 'r'))+json.load(open(__C.ANN_PATH['refcoco'], 'r'))

        self.ques_list = []
        splits=split.split('+')
        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]


        refs=[]

        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)
        for split in total_refs_list:
            for ann in total_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)



        self.image_path=__C.IMAGE_PATH[__C.DATASET]
        self.mask_path=__C.MASK_PATH[__C.DATASET]
        self.input_shape=__C.INPUT_SHAPE
        self.flip_lr=__C.FLIP_LR if split=='train' else False
        # Define run data size
        self.data_size = len(self.refs_anno)

        print(' ========== Dataset size:', self.data_size)
        # ------------------------
        # ---- Data statistic ----
        # ------------------------
        # Tokenize
        self.token_to_ix,self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(stat_refs_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        self.max_token = __C.MAX_TOKEN
        if self.max_token == -1:
            self.max_token = max_token
        print('Max token length:', max_token, 'Trimmed to:', self.max_token)
        print('Finished!')
        print('')

        self.candidate_transforms ={}
        if  self.split == 'train':
            if 'RandAugment' in self.__C.DATA_AUGMENTATION:
                self.candidate_transforms['RandAugment']=RandAugment(2,9)
            if 'ElasticTransform' in self.__C.DATA_AUGMENTATION:
                self.candidate_transforms['ElasticTransform']=A.ElasticTransform(p=0.5)
            if 'GridDistortion' in self.__C.DATA_AUGMENTATION:
                self.candidate_transforms['GridDistortion']=A.GridDistortion(p=0.5)
            if 'RandomErasing' in self.__C.DATA_AUGMENTATION:
                self.candidate_transforms['RandomErasing']=transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8),
                                                                              value="random")
        self.transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize(__C.MEAN, __C.STD)])



    def tokenize(self, stat_refs_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        max_token = 0
        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    words = re.sub(
                        r"([.,'!?\"()*#:;])",
                        '',
                        ref.lower()
                    ).replace('-', ' ').replace('/', ' ').split()

                    if len(words) > max_token:
                        max_token = len(words)

                    for word in words:
                        if word not in token_to_ix:
                            token_to_ix[word] = len(token_to_ix)
                            if use_glove:
                                pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        ix_to_token={}
        for item in token_to_ix:
            ix_to_token[token_to_ix[item]]=item

        return token_to_ix, ix_to_token,pretrained_emb, max_token


    def proc_ref(self, ref, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ref.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_refs(self, idx):
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        ref=self.proc_ref(ref,self.token_to_ix,self.max_token)
        return ref

    def preprocess_info(self,img,mask,box,iid,lr_flip=False):
        h, w, _ = img.shape
        # img = img[:, :, ::-1]
        imgsize=self.input_shape[0]
        new_ar = w / h
        if new_ar < 1:
            nh = imgsize
            nw = nh * new_ar
        else:
            nw = imgsize
            nh = nw / new_ar
        nw, nh = int(nw), int(nh)


        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

        img = cv2.resize(img, (nw, nh))
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = img
        info_img = (h, w, nh, nw, dx, dy,iid)

        mask=np.expand_dims(mask,-1).astype(np.float32)
        mask=cv2.resize(mask, (nw, nh))
        mask=np.expand_dims(mask,-1).astype(np.float32)
        sized_mask = np.zeros((imgsize, imgsize, 1), dtype=np.float32)
        sized_mask[dy:dy + nh, dx:dx + nw, :]=mask
        sized_mask=np.transpose(sized_mask, (2, 0, 1))
        sized_box=label2yolobox(box,info_img,self.input_shape[0],lrflip=lr_flip)
        return sized,sized_mask,sized_box, info_img

    def load_img_feats(self, idx):
        img_path=None
        if self.__C.DATASET in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join(self.image_path,'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.__C.DATASET=='referit':
            img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
        elif self.__C.DATASET=='vg':
            img_path = os.path.join(self.image_path, self.refs_anno[idx]['url'])
        elif self.__C.DATASET == 'merge':
            if self.refs_anno[idx]['data_source']=='coco':
                iid='COCO_train2014_%012d.jpg'%int(self.refs_anno[idx]['iid'].split('.')[0])
            else:
                iid=self.refs_anno[idx]['iid']
            img_path = os.path.join(self.image_path,self.refs_anno[idx]['data_source'], iid)
        else:
            assert NotImplementedError

        image= cv2.imread(img_path)

        if self.__C.DATASET in ['refcoco','refcoco+','refcocog','referit']:
            mask=np.load(os.path.join(self.mask_path,'%d.npy'%self.refs_anno[idx]['mask_id']))
        else:
            mask=np.zeros([image.shape[0],image.shape[1],1],dtype=np.float)

        box=np.array([self.refs_anno[idx]['bbox']])

        return image,mask,box,self.refs_anno[idx]['mask_id'],self.refs_anno[idx]['iid']

    def __getitem__(self, idx):

        ref_iter = self.load_refs(idx)
        image_iter,mask_iter,gt_box_iter,mask_id,iid= self.load_img_feats(idx)
        image_iter = cv2.cvtColor(image_iter, cv2.COLOR_BGR2RGB)
        ops=None
        if len(list(self.candidate_transforms.keys()))>0:
            ops = random.choices(list(self.candidate_transforms.keys()), k=1)[0]
        if ops is not None and ops!='RandomErasing':
            image_iter = self.candidate_transforms[ops](image=image_iter)['image']

        flip_box=False
        if self.flip_lr and random.random() < 0.5:
            image_iter=image_iter[::-1]
            flip_box=True
        image_iter, mask_iter, box_iter,info_iter=self.preprocess_info(image_iter,mask_iter,gt_box_iter.copy(),iid,flip_box)
        return \
            torch.from_numpy(ref_iter).long(), \
            self.transforms(image_iter),\
            torch.from_numpy(mask_iter).float(),\
            torch.from_numpy(box_iter).float(), \
            torch.from_numpy(gt_box_iter).float(), \
            mask_id,\
            np.array(info_iter)

    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)

def loader(__C,dataset: torch.utils.data.Dataset, rank: int, shuffle,drop_last=False):
    if __C.MULTIPROCESSING_DISTRIBUTED:
        assert __C.BATCH_SIZE % len(__C.GPU) == 0
        assert __C.NUM_WORKER % len(__C.GPU) == 0
        assert dist.is_initialized()

        dist_sampler = DistributedSampler(dataset,
                                          num_replicas=__C.WORLD_SIZE,
                                          rank=rank)

        data_loader = DataLoader(dataset,
                                 batch_size=__C.BATCH_SIZE // len(__C.GPU),
                                 shuffle=shuffle,
                                 sampler=dist_sampler,
                                 num_workers=__C.NUM_WORKER //len(__C.GPU),
                                 pin_memory=True,
                                 drop_last=drop_last)  # ,
                                # prefetch_factor=_C['PREFETCH_FACTOR'])  only works in PyTorch 1.7.0
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=__C.BATCH_SIZE,
                                 shuffle=shuffle,
                                 num_workers=__C.NUM_WORKER,
                                 pin_memory=True,
                                 drop_last=drop_last)
    return data_loader

if __name__ == '__main__':

    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.ANN_PATH= {
                'refcoco': './data/anns/refcoco.json',
                'refcoco+': './data/anns/refcoco+.json',
                'refcocog': './data/anns/refcocog.json',
                'vg': './data/anns/vg.json',
            }

            self.IMAGE_PATH={
                'refcoco': './data/images/train2014',
                'refcoco+': './data/images/train2014',
                'refcocog': './data/images/train2014',
                'vg': './data/images/VG'
            }

            self.MASK_PATH={
                'refcoco': './data/masks/refcoco',
                'refcoco+': './data/masks/refcoco+',
                'refcocog': './data/masks/refcocog',
                'vg': './data/masks/vg'}
            self.INPUT_SHAPE = (416, 416)
            self.USE_GLOVE = True
            self.DATASET = 'vg'
            self.MAX_TOKEN = 15
            self.MEAN = [0., 0., 0.]
            self.STD = [1., 1., 1.]
    cfg=Cfg()
    dataset=RefCOCODataSet(cfg,'val')
    data_loader = DataLoader(dataset,
                             batch_size=10,
                             shuffle=False,
                             pin_memory=True)
    for _,ref_iter,image_iter, mask_iter, box_iter,words,info_iter in data_loader:
        print(ref_iter)
        # print(image_iter.size())
        # print(mask_iter.size())
        # print(box_iter.size())
        # print(ref_iter.size())
        # # cv2.imwrite('./test.jpg', image_iter.numpy()[0].transpose((1, 2, 0))*255)
        # # cv2.imwrite('./mask.jpg', mask_iter.numpy()[0].transpose((1, 2, 0))*255)
        # print(info_iter.size())
        # print(info_iter)





