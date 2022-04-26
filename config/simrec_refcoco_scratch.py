from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.simrec import model

# Refine data path depend your own need
dataset.ann_path["refcoco"] = "/home/rentianhe/dataset/rec/anns/refcoco.json"
dataset.image_path["refcoco"] = "/home/rentianhe/dataset/rec/images/train2014"
dataset.mask_path["refcoco"] = "/home/rentianhe/dataset/rec/masks/refcoco"

# # Refine training cfg
train.output_dir = "./output/det_base_refcoco_baseline"
train.resume_path = ""
train.batch_size = 4
train.save_period = 2

# Refine model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.pretrained_weight_path="./data/weights/cspdarknet_coco.pth"
