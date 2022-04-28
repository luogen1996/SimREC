from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.simrec import model


# # Refine training cfg
train.output_dir = "./output"
train.batch_size = 32
train.save_period = 1
train.log_period = 10
train.evaluation.eval_batch_size = 32

# Refine model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.pretrained_weight_path="./data/weights/cspdarknet_coco.pth"
