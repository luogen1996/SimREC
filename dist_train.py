import os
import time
import datetime
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from simrec.config import LazyConfig, instantiate
from simrec.datasets.dataloader import build_loader
from simrec.scheduler.build import build_lr_scheduler
from simrec.utils.metric import AverageMeter, ProgressMeter
from simrec.utils.distributed import reduce_meters, main_process, cleanup_distributed
from simrec.utils.env import seed_everything, setup_unique_version
from simrec.utils.model_ema import EMA
from simrec.utils.logger import create_logger


from test import validate


def train_one_epoch(cfg, model, optimizer, scheduler, data_loader, scalar, writer, epoch, rank, ema=None):
    model.train()
    data_loader.sampler.set_epoch(epoch)
    
    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.7f')
    losses_det = AverageMeter('LossDet', ':.7f')
    losses_seg = AverageMeter('LossSeg', ':.7f')
    meters = [batch_time, data_time, losses, losses_det, losses_seg]
    meters_dict = {meter.name: meter for meter in meters}
    
    start = time.time()
    end = time.time()
    for idx, (ref_iter, image_iter, mask_iter, box_iter, gt_box_iter, mask_id, info_iter) in enumerate(data_loader):
        data_time.update(time.time() - end)
        
        ref_iter = ref_iter.cuda(non_blocking=True)
        image_iter = image_iter.cuda(non_blocking=True)
        mask_iter = mask_iter.cuda(non_blocking=True)
        box_iter = box_iter.cuda( non_blocking=True)

        if cfg.train.multi_scale_training:
            img_scales = cfg.train.multi_scale_training.img_scales
            h, w = img_scales[np.random.randint(0, len(img_scales))]
            image_iter = F.interpolate(image_iter, (h, w))
            mask_iter = F.interpolate(mask_iter, (h, w))

        if scalar is not None:
            with torch.cuda.amp.autocast():
                loss, loss_det, loss_seg = model(image_iter,ref_iter,det_label=box_iter,seg_label=mask_iter)
        else:
            loss, loss_det, loss_seg = model(image_iter, ref_iter, det_label=box_iter,seg_label=mask_iter)

        optimizer.zero_grad()
        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            scalar.update()
        else:
            loss.backward()
            # for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            #     print(p.grad.data)
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            optimizer.step()
        scheduler.step()
        
        if ema is not None:
            ema.update_params()
        
        losses.update(loss.item(), image_iter.size(0))
        losses_det.update(loss_det.item(), image_iter.size(0))
        losses_seg.update(loss_seg.item(), image_iter.size(0))

        reduce_meters(meters_dict, rank, cfg)
        if dist.get_rank() == 0:
            global_step = epoch * num_iters + idx
            writer.add_scalar("loss/train", losses.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_det/train", losses_det.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_seg/train", losses_seg.avg_reduce, global_step=global_step)
        
        if idx % cfg.train.log_period == 0 or idx==len(data_loader):
            # progress.display(epoch, ith_batch)
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_iters - idx)
            logger.info(
                f'Train: [{epoch}/{cfg.train.epochs}][{idx}/{num_iters}\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.7f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'det loss {losses_det.val:.4f} ({losses_det.avg:.4f})\t'
                f'seg loss {losses_seg.val:.4f} ({losses_seg.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
        
        # break
        batch_time.update(time.time() - end)
        end = time.time()
    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")



def main(cfg):
    global best_det_acc,best_seg_acc
    best_det_acc,best_seg_acc=0.,0.

    # build training dataset and dataloader
    cfg.dataset.split = "train"
    train_set = instantiate(cfg.dataset)
    train_loader = build_loader(
        cfg, 
        train_set, 
        dist.get_rank(), 
        shuffle=True,
        drop_last=True
    )
    
    # build validation dataset and dataloader
    cfg.dataset.split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader = build_loader(
        cfg, 
        val_set,
        dist.get_rank(),
        shuffle=False
    )

    # build model
    cfg.model.language_encoder.pretrained_emb = train_set.pretrained_emb
    cfg.model.language_encoder.token_size = train_set.token_size
    net = instantiate(cfg.model)

    # build optimizer
    params = filter(lambda p: p.requires_grad, net.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)

    # model ema
    ema=None

    torch.cuda.set_device(dist.get_rank())
    net = DistributedDataParallel(net.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)


    if main_process(dist.get_rank()):
        print(cfg)
        print(net)
        total = sum([param.nelement() for param in net.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))
        total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6))

    cfg.train.scheduler.epochs = cfg.train.epochs
    cfg.train.scheduler.n_iter_per_epoch = len(train_loader)
    scheduler = build_lr_scheduler(cfg.train.scheduler, optimizer)

    start_epoch = 0

    if os.path.isfile(cfg.train.resume_path):
        checkpoint = torch.load(
            cfg.train.resume_path, 
            map_location=lambda storage, 
            loc: storage.cuda()
        )
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys())==0:
            new_dict=checkpoint['state_dict']
        net.load_state_dict(new_dict,strict=False)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        if main_process(dist.get_rank()):
            print("==> loaded checkpoint from {}\n".format(cfg.train.resume_path) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))

    if os.path.isfile(cfg.train.vl_pretrain_weight):
        checkpoint = torch.load(cfg.train.vl_pretrain_weight, map_location=lambda storage, loc: storage.cuda() )
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys())==0:
            new_dict=checkpoint['state_dict']
        net.load_state_dict(new_dict,strict=False)
        start_epoch = 0
        if main_process(dist.get_rank()):
            print("==> loaded checkpoint from {}\n".format(cfg.train.vl_pretrain_weight) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'], checkpoint['lr']))



    if cfg.train.amp:
        assert torch.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    if main_process(dist.get_rank()):
        writer = SummaryWriter(log_dir=os.path.join(cfg.train.log_path, str(cfg.train.version)))
    else:
        writer = None

    save_ids=np.random.randint(1, len(val_loader) * cfg.train.batch_size, 100) if cfg.train.log_image else None

    for ith_epoch in range(start_epoch, cfg.train.epochs):
        if cfg.train.use_ema and ema is None:
            ema = EMA(net, 0.9997)
        train_one_epoch(cfg, net, optimizer,scheduler,train_loader,scalar,writer,ith_epoch,dist.get_rank(),ema)
        box_ap,mask_ap=validate(cfg, net,val_loader, writer,ith_epoch,dist.get_rank(),val_set.ix_to_token,save_ids=save_ids,ema=ema)

        if main_process(dist.get_rank()):
            if ema is not None:
                ema.apply_shadow()
            torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),'lr':optimizer.param_groups[0]["lr"],},
                       os.path.join(cfg.train.log_path, str(cfg.train.version),'ckpt', 'last.pth'))
            if box_ap>best_det_acc:
                best_det_acc=box_ap
                torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),'lr':optimizer.param_groups[0]["lr"],},
                           os.path.join(cfg.train.log_path, str(cfg.train.version),'ckpt', 'det_best.pth'))
            if mask_ap > best_seg_acc:
                best_seg_acc=mask_ap
                torch.save({'epoch': ith_epoch + 1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),'lr':optimizer.param_groups[0]["lr"],},
                           os.path.join(cfg.train.log_path, str(cfg.train.version),'ckpt', 'seg_best.pth'))
            if ema is not None:
                ema.restore()

    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimREC")
    parser.add_argument('--config', type=str, required=True, default='./config/simrec_refcoco_scratch.py')
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # Environments setting
    setup_unique_version(cfg)
    seed_everything(cfg.train.seed)

    # Distributed setting
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=cfg.train.ddp.backend, 
        init_method=cfg.train.ddp.init_method, 
        world_size=world_size, 
        rank=rank
    )
    torch.distributed.barrier()

    # Path setting
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=cfg.train.output_dir, dist_rank=dist.get_rank(), name=f"{cfg.model._target_}")
    
    # Logger setting
    if not os.path.exists(os.path.join(cfg.train.log_path, str(cfg.train.version))):
        os.makedirs(os.path.join(cfg.train.log_path, str(cfg.train.version),'ckpt'), exist_ok=True)

    if dist.get_rank() == 0:
        path = os.path.join(cfg.train.output_dir, "config.yaml")
        LazyConfig.save(cfg, path)

    main(cfg)
