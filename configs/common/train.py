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