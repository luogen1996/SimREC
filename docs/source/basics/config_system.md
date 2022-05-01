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
