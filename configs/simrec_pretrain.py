from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.simrec import model

# Refine training cfg
train.epochs = 15
train.batch_size = 32
train.output_dir = "./output/det_base_refcoco_vg_baseline"

# Refine dataset cfg
dataset.dataset = "merge"
dataset.max_token_length = 15

# Refine optim
optim.lr = train.base_lr

# Refine model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.freeze_backbone = False
