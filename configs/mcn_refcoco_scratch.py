from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.mcn import model

# # Refine training cfg
train.output_dir = "./output/mcn_refcoco_scratch"
train.batch_size = 32
train.save_period = 1
train.log_period = 10
train.evaluation.eval_batch_size = 32
train.epochs = 39
train.scheduler.name = "step"
train.ema.enabled = False
train.multi_scale_training.enabled = False

# Refine optim
optim.lr = train.base_lr


# Refine model cfg
model.visual_backbone.pretrained = False
model.visual_backbone.pretrained_weight_path="./data/weights/darknet_coco.pth"
