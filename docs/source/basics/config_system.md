## SimREC Config
We adopted the `Lazy Config System` design from detectron2, You can refer to [d2 tutorials](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) for more details aboud the syntax and usage of lazy config. In this section, we provide the basic examples of the lazy config usage in SimREC.

### Configs in SimREC
We simply summarized the config namespaces into `model`, `dataset`, `optim`, `train` as follows:

#### model
This is the configuration for the model definition. You can refer to `configs/common/models` for more examples:

Here is the example of loading `SimREC` model config file:
```python
# configs/common/models/simrec.py
import torch.nn as nn

from simrec.config import LazyCall
from simrec.models.simrec import SimREC
from simrec.models.backbones import CspDarkNet
from simrec.models.heads import REChead
from simrec.models.language_encoders import LSTM_SA
from simrec.layers.fusion_layer import SimpleFusion, MultiScaleFusion, GaranAttention

model = LazyCall(SimREC)(
    visual_backbone=LazyCall(CspDarkNet)(
        pretrained=False,
        pretrained_weight_path="./data/weights/cspdarknet_coco.pth",
        freeze_backbone=True,
        multi_scale_outputs=True,
    ),
    language_encoder=LazyCall(LSTM_SA)(
        depth=1,
        hidden_size=512,
        num_heads=8,
        ffn_size=2048,
        flat_glimpses=1,
        word_embed_size=300,
        dropout_rate=0.1,
        # language_encoder.pretrained_emb and language.token_size is meant to be set
        # before instantiating
        freeze_embedding=True,
        use_glove=True,
    ),
    multi_scale_manner=LazyCall(MultiScaleFusion)(
        v_planes=(512, 512, 512),
        scaled=True
    ),
    fusion_manner=LazyCall(nn.ModuleList)(
        modules = [
            LazyCall(SimpleFusion)(v_planes=256, out_planes=512, q_planes=512),
            LazyCall(SimpleFusion)(v_planes=512, out_planes=512, q_planes=512),
            LazyCall(SimpleFusion)(v_planes=1024, out_planes=512, q_planes=512),
        ]
    ),
    attention_manner=LazyCall(GaranAttention)(
        d_q=512,
        d_v=512
    ),
    head=LazyCall(REChead)(
        label_smooth=0.,
        num_classes=0,
        width=1.0,
        strides=[32,],
        in_channels=[512,],
        act="silu",
        depthwise=False
    )
)
```
```python
# my_own_config_file.py
from common.models.simrec import model

model.visual_backbone.pretrained = True # modify config according to your own needs
```
**Note:** due to the lazy instantiate benefits of LazyCall, some param like `model.language_encoder.pretrained_emb` will be set in train loop before instantiating the model.

#### train
This is the configuration for training. The default dataset config can be found in `configs/common/train.py`. The details of training config are as follows:
```python
# Basic training-related configs.

train = dict(
    
    # Directory where to save the output files.
    output_dir = "./test",
    
    # Warmup epochs and total epochs for training.
    warmup_epochs=3,
    epochs = 25,

    # Learning rate settings for lr-scheduler.
    base_lr=1e-4,
    warmup_lr=1e-7,
    min_lr=1e-6,

    # Total batch size, if you run SimREC on 4 GPUs,
    # each gpu will handle only (batch / 4) samples.
    batch_size=8,

    # Evaluation configuration, if set sequential=True, which will
    # use SequentialSampler during validating.
    evaluation=dict(
        eval_batch_size=8, 
        sequential=False
    ),

    # Log the training infos every log_period times of iterations.
    log_period=1,

    # Save the checkpoints every save_period times of iterations.
    save_period=1,

    # Basic data config.
    data=dict(
        pin_memory=True, 
        num_workers=8,
        mean=[0., 0., 0.],
        std=[1., 1., 1.],
    ),

    # Scheduler config.
    scheduler=dict(
        name="cosine",
        decay_epochs=[30, 35, 37],
        lr_decay_rate=0.2,
    ),

    # Enable automatic mixed precision for training which does not 
    # change model's inference behavior.
    amp=dict(enabled=False),

    # Distributed training settings.
    ddp=dict(
        backend="nccl",
        init_method="env://",
    ),

    # Enable model ema during training or not.
    ema=dict(enabled=True, alpha=0.9997, buffer_ema=True),

    # Automatically convert batchnorm to sync batchnorm layers.
    sync_bn=dict(enabled=False),

    # Clip gradient.
    clip_grad_norm=0.15,

    # Resume training.
    auto_resume=dict(enabled=True),
    resume_path="",
    vl_pretrain_weight="",

    # Multi-scale training.
    multi_scale_training=dict(
        enabled=True,
        img_scales=[[224,224],[256,256],[288,288],[320,320],[352,352],
                    [384,384],[416,416],[448,448],[480,480],[512,512],
                    [544,544],[576,576],[608,608]]
    ),

    # Log image during training.
    log_image=False,

    # Training seed.
    seed = 123456,
)
```

#### optim
This is the configuration for the optim definition. Please refer to `configs/common/optim.py` for the default optim config:
```python
from torch.optim import Adam

from simrec.config import LazyCall

optim = LazyCall(Adam)(
    # optim.params is meant to be set before instantiating
    lr=0.0001,
    betas=(0.9, 0.98),
    eps=1e-9
)
```

#### dataset
This is the configuration for dataset. The default dataset config can be found in `configs/common/dataset.py`. The details of dataset config are as follows:
```python
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
```


#### Use the default config in your own config file
The users do not have to rewrite all the config every time. You can use the default config file provided in SimREC by importing them as the python file. For example:
```python
# import the default config
from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.simrec import model

# modify them according to your own needs
# refine the cfg
train.output_dir = "./your/own/path"
train.batch_size = 32
train.save_period = 1
train.log_period = 10
train.evaluation.eval_batch_size = 32
train.sync_bn.enabled = False
```