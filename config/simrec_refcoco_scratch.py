from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.simrec import model


dataset.ann_path["refcoco"] = "/home/rentianhe/dataset/rec/anns/refcoco.json"
dataset.image_path["refcoco"] = "/home/rentianhe/dataset/rec/images/train2014"
dataset.mask_path["refcoco"] = "/home/rentianhe/dataset/rec/masks/refcoco"