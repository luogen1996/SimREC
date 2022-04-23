import random

from simrec.config import LazyCall
from simrec.scheduler.lr_scheduler import WarmupCosineLR

train = dict(
    gpus = [0],
    num_workers = 8,
    tag = "test",
    # Directory where output files are written
    output_dir = "./output",
    version = random.randint(0, 99999),
    log_path = "./logs/det_base_refcoco_baseline",
    resume_path = "",
    batch_size = 4,
    vl_pretrain_weight="",
    epochs = 25,

    print_freq = 100,

    scheduler = LazyCall(WarmupCosineLR)(
        # optimizer and epochs and n_iter_per_epoch will be set in train.py
        warmup_epochs = 3,
        warmup_lr = 0.0000001,
        base_lr = 0.0001,
        min_lr = 0.000001,
    ),

    distributed = dict(
        enabled = True,
        node_id = 0,
        world_size = 1,
        dist_url = "tcp://127.0.0.1:12345",
        multiprocessing_distributed = True,
        rank = 0
    ),

    multi_scale = [[224,224],[256,256],[288,288],[320,320],[352,352],
                   [384,384],[416,416],[448,448],[480,480],[512,512],
                   [544,544],[576,576],[608,608]],

    grad_norm_clip = 0.15,
    use_ema = True,
    amp = False,
    log_image = False,
    seed = 123456,
)