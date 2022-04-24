import random

from simrec.config import LazyCall
from simrec.scheduler.lr_scheduler import WarmupCosineLR

train = dict(
    num_workers = 8,
    amp=dict(enabled=False),
    ddp=dict(
        backend="nccl",
        init_method="env://",
        rank=0
    ),
    ema=dict(enabled=True),
    epochs = 25,
    output_dir = "./output",
    log_period = 100,
    version = random.randint(0, 99999),
    log_path = "./logs/det_base_refcoco_baseline",
    resume_path = "",
    batch_size = 4,
    vl_pretrain_weight="",

    print_freq = 100,

    scheduler = LazyCall(WarmupCosineLR)(
        # optimizer and epochs and n_iter_per_epoch will be set in train.py
        warmup_epochs = 3,
        warmup_lr = 0.0000001,
        base_lr = 0.0001,
        min_lr = 0.000001,
    ),

    multi_scale_training=dict(
        enabled=True,
        img_scales=[[224,224],[256,256],[288,288],[320,320],[352,352],
                    [384,384],[416,416],[448,448],[480,480],[512,512],
                    [544,544],[576,576],[608,608]]
    ),

    clip_grad_norm=0.15,
    log_image = False,
    seed = 123456,
)