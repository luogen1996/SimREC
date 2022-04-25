import random

from simrec.config import LazyCall
from simrec.scheduler.lr_scheduler import WarmupCosineLR

train = dict(
    output_dir = "./test",
    warmup_epochs=3,
    epochs = 25,
    base_lr=1e-4,
    warmup_lr=1e-7,
    min_lr=1e-6,
    batch_size=8,
    log_period=1,
    data=dict(pin_memory=True, num_workers=8),
    scheduler=dict(
        name="cosine",
        decay_epochs=[30, 35, 37],
        lr_decay_rate=0.2,
    )
    amp=dict(enabled=False),
    ddp=dict(
        backend="nccl",
        init_method="env://",
    ),
    ema=dict(enabled=True, alpha=0.9997, buffer_ema=True),
    resume=dict(enable=False, auto_resume=True, resume_path=""),
    vl_pretrain_weight="",

    # scheduler = LazyCall(WarmupCosineLR)(
    #     # optimizer and epochs and n_iter_per_epoch will be set in train.py
    #     warmup_epochs = 3,
    #     warmup_lr = 0.0000001,
    #     base_lr = 0.0001,
    #     min_lr = 0.000001,
    # ),

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