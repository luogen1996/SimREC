import albumentations as A
from torchvision.transforms import transforms

from simrec.config import LazyCall
from simrec.datasets.dataset import RefCOCODataSet
from simrec.datasets.transforms.randaug import RandAugment

from .train import train

dataset = LazyCall(RefCOCODataSet)(
    # the dataset to be created
    # choose from ["refcoco", "refcoco+", "refcocog", "referit", "vg", "merge"]
    dataset = "refcoco",

    # path to the files
    ann_path = {
                'refcoco':'./data/anns/refcoco.json',
                'refcoco+': './data/anns/refcoco+.json',
                'refcocog': './data/anns/refcocog.json',
                'referit': './data/anns/refclef.json',
                'flickr': './data/anns/flickr.json',
                'vg': './data/anns/vg.json',
                'merge':'./data/anns/merge.json'
            },
    image_path = {
                'refcoco': './data/images/coco',
                'refcoco+': './data/images/coco',
                'refcocog': './data/images/coco',
                'referit': './data/images/refclef',
                'flickr': './data/images/flickr',
                'vg':'./data/images/VG',
                'merge':'./data/images/'
            },
    mask_path = {
                'refcoco': './data/masks/refcoco',
                'refcoco+': './data/masks/refcoco+',
                'refcocog': './data/masks/refcocog',
                'referit': './data/masks/refclef'
            },
    
    # original input image shape
    input_shape = [416, 416],
    flip_lr = False,

    # basic transforms
    transforms=LazyCall(transforms.Compose)(
        transforms = [
            LazyCall(transforms.ToTensor)(),
            LazyCall(transforms.Normalize)(
                mean=train.data.mean, 
                std=train.data.std,
            )
        ]
    ),

    # candidate transforms
    candidate_transforms = {
        # "RandAugment": RandAugment(2, 9),
        # "ElasticTransform": A.ElasticTransform(p=0.5),
        # "GridDistortion": A.GridDistortion(p=0.5),
        # "RandomErasing": transforms.RandomErasing(
        #     p = 0.3,
        #     scale = (0.02, 0.2),
        #     ratio=(0.05, 8),
        #     value="random",
        # )
    },

    # the max truncked length for language tokens
    max_token_length = 15,

    # use glove pretrained embeddings or not
    use_glove = True,

    # datasets splits
    split = "train",
)