from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.mcn import model

# Refine data path depend your own need
dataset.ann_path["refcoco"] = "/home/rentianhe/dataset/rec/anns/refcoco.json"
dataset.image_path["refcoco"] = "/home/rentianhe/dataset/rec/images/train2014"
dataset.mask_path["refcoco"] = "/home/rentianhe/dataset/rec/masks/refcoco"

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
train.sync_bn.enabled = True

# Refine optim
optim.lr = train.base_lr


# Refine model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.pretrained_weight_path="./data/weights/darknet_coco.weights"
