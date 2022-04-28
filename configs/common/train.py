# Basic training-related configs 

train = dict(
    output_dir = "./test",
    warmup_epochs=3,
    epochs = 25,
    base_lr=1e-4,
    warmup_lr=1e-7,
    min_lr=1e-6,
    batch_size=8,
    evaluation=dict(
        eval_batch_size=8, 
        sequential=False
    ),
    log_period=1,
    save_period=1,
    data=dict(
        pin_memory=True, 
        num_workers=8,
        mean=[0., 0., 0.],
        std=[1., 1., 1.],
    ),
    scheduler=dict(
        name="cosine",
        decay_epochs=[30, 35, 37],
        lr_decay_rate=0.2,
    ),
    amp=dict(enabled=False),
    ddp=dict(
        backend="nccl",
        init_method="env://",
    ),
    ema=dict(enabled=True, alpha=0.9997, buffer_ema=True),
    sync_bn=dict(enabled=False),
    clip_grad_norm=0.15,
    auto_resume=dict(enabled=True),
    resume_path="",
    vl_pretrain_weight="",
    multi_scale_training=dict(
        enabled=True,
        img_scales=[[224,224],[256,256],[288,288],[320,320],[352,352],
                    [384,384],[416,416],[448,448],[480,480],[512,512],
                    [544,544],[576,576],[608,608]]
    ),
    log_image=False,
    seed = 123456,
)