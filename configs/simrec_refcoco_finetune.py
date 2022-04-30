from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.simrec import model

# Refine training cfg
train.batch_size = 4
train.output_dir = "./output/det_base_refcoco_finetune"

# Refine optim
optim.lr = train.base_lr

# Refine model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.pretrained_weight_path = "./data/weights/SimREC_pretrain_merge.pth"