import os
import sys
import time
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from simrec.config import instantiate, LazyConfig
from simrec.datasets.dataloader import build_loader
from simrec.datasets.utils import yolobox2label
from simrec.models.utils import batch_box_iou, mask_processing, mask_iou
from simrec.utils.distributed import is_main_process, reduce_meters
from simrec.utils.visualize import draw_visualization, normed2original
from simrec.utils.env import seed_everything, setup_unique_version
from simrec.utils.logger import create_logger
from simrec.utils.metric import AverageMeter


def validate(cfg, model, data_loader, writer, epoch, ix_to_token, logger, rank, save_ids=None, prefix='Val', ema=None):
    if ema is not None:
        ema.apply_shadow()
    model.eval()

    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')
    mask_ap = AverageMeter('MaskIoU', ':6.2f')
    inconsistency_error = AverageMeter('IE', ':6.2f')
    mask_aps={}
    for item in np.arange(0.5, 1, 0.05):
        mask_aps[item]=[]
    meters = [batch_time, data_time, losses, box_ap, mask_ap,inconsistency_error]
    meters_dict = {meter.name: meter for meter in meters}
    
    with torch.no_grad():
        end = time.time()
        for idx, (ref_iter, image_iter, mask_iter, box_iter,gt_box_iter, mask_id, info_iter) in enumerate(data_loader):
            ref_iter = ref_iter.cuda( non_blocking=True)
            image_iter = image_iter.cuda( non_blocking=True)
            box_iter = box_iter.cuda( non_blocking=True)
            box, mask= model(image_iter, ref_iter)

            gt_box_iter=gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])
            gt_box_iter=gt_box_iter.cpu().numpy()
            info_iter=info_iter.cpu().numpy()
            box=box.squeeze(1).cpu().numpy()
            pred_box_vis=box.copy()

            # predictions to ground-truth
            for i in range(len(gt_box_iter)):
                box[i]=yolobox2label(box[i],info_iter[i])


            box_iou=batch_box_iou(torch.from_numpy(gt_box_iter),torch.from_numpy(box)).cpu().numpy()
            seg_iou=[]
            mask=mask.cpu().numpy()
            for i, mask_pred in enumerate(mask):
                if writer is not None and save_ids is not None and idx * cfg.train.BATCH_SIZE+i in save_ids:
                    ixs=ref_iter[i].cpu().numpy()
                    words=[]
                    for ix in ixs:
                        if ix >0:
                            words.append(ix_to_token[ix])
                    sent=' '.join(words)
                    box_iter = box_iter.view(box_iter.shape[0], -1) * cfg.dataset.input_shape[0]
                    box_iter[:, 0] = box_iter[:, 0] - 0.5 * box_iter[:, 2]
                    box_iter[:, 1] = box_iter[:, 1] - 0.5 * box_iter[:, 3]
                    box_iter[:, 2] = box_iter[:, 0] + box_iter[:, 2]
                    box_iter[:, 3] = box_iter[:, 1] + box_iter[:, 3]
                    det_image = draw_visualization(
                        image=normed2original(image_iter[i], cfg.train.data.mean, cfg.train.data.std),
                        sent=sent,
                        pred_box=pred_box_vis[i].cpu().numpy(),
                        gt_box=box_iter[i].cpu().numpy()
                    )
                    writer.add_image('image/' + str(idx * cfg.train.batch_size + i) + '_det',det_image,epoch,dataformats='HWC')
                    writer.add_image('image/' + str(idx * cfg.train.batch_size + i) + '_seg', (mask[i,None]*255).astype(np.uint8))

                # from pydensecrf import densecrf
                # d = densecrf.DenseCRF2D(416, 416, 2)
                # U = np.expand_dims(-np.log(mask_pred), axis=0)
                # U_ = np.expand_dims(-np.log(1 - mask_pred), axis=0)
                # unary = np.concatenate((U_, U), axis=0)
                # unary = unary.reshape((2, -1))
                # d.setUnaryEnergy(unary)
                # d.addPairwiseGaussian(sxy=4, compat=3)
                # d.addPairwiseBilateral(sxy=26, srgb=3, rgbim=np.ascontiguousarray((image_iter[i].cpu().numpy()*255).astype(np.uint8).transpose([1,2,0])), compat=10)
                # Q = d.inference(5)
                # mask_pred = np.argmax(Q, axis=0).reshape((416, 416)).astype(np.float32)


                mask_gt=np.load(os.path.join(cfg.dataset.mask_path[cfg.dataset.dataset],'%d.npy' % mask_id[i]))
                mask_pred=mask_processing(mask_pred,info_iter[i])

                single_seg_iou,single_seg_ap=mask_iou(mask_gt,mask_pred)
                for item in np.arange(0.5, 1, 0.05):
                    mask_aps[item].append(single_seg_ap[item]*100.)
                seg_iou.append(single_seg_iou)
            seg_iou=np.array(seg_iou).astype(np.float32)

            ie=(box_iou>=0.5).astype(np.float32)*(seg_iou<0.5).astype(np.float32)+(box_iou<0.5).astype(np.float32)*(seg_iou>=0.5).astype(np.float32)
            inconsistency_error.update(ie.mean()*100., ie.shape[0])
            box_ap.update((box_iou>0.5).astype(np.float32).mean()*100., box_iou.shape[0])
            mask_ap.update(seg_iou.mean()*100., seg_iou.shape[0])

            reduce_meters(meters_dict, rank, cfg)

            if (idx % cfg.train.log_period == 0 or idx==(len(data_loader)-1)):
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Evaluation on {prefix}: [{idx}/{len(data_loader)}]  '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                    f'BoxIoU@0.5 {box_ap.val:.4f} ({box_ap.avg:.4f})  '
                    f'MaskIoU {mask_ap.val:.4f} ({mask_ap.avg:.4f})  '
                    f'IE {inconsistency_error.val:.4f} ({inconsistency_error.avg:.4f})  '
                    f'Mem {memory_used:.0f}MB')
            batch_time.update(time.time() - end)
            end = time.time()

        if is_main_process() and writer is not None:
            writer.add_scalar("Acc/BoxIoU@0.5", box_ap.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/MaskIoU", mask_ap.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/IE", inconsistency_error.avg_reduce, global_step=epoch)
            for item in mask_aps:
                writer.add_scalar("Acc/MaskIoU@%.2f"%item, np.array(mask_aps[item]).mean(), global_step=epoch)

        logger.info(f' * BoxIoU@0.5 {box_ap.avg:.3f} MaskIoU {mask_ap.avg:.3f}')

    if ema is not None:
        ema.restore()
    return box_ap.avg_reduce, mask_ap.avg_reduce


def main(cfg):
    global best_det_acc,best_seg_acc
    best_det_acc,best_seg_acc=0.,0.

    # build training dataset and dataloader
    cfg.dataset.split = "train"
    train_set = instantiate(cfg.dataset)
    train_loader=build_loader(
        cfg,
        train_set, 
        dist.get_rank(),
        shuffle=True
    )

    # build single or multi-datasets for validation
    loaders=[]
    prefixs=['val']
    cfg.dataset.split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader=build_loader(
        cfg,
        val_set,
        dist.get_rank(),
        shuffle=False
    )
    loaders.append(val_loader)
    
    if cfg.dataset.dataset == 'refcoco' or cfg.dataset.dataset == 'refcoco+':
        cfg.dataset.split = "testA"
        testA_dataset = instantiate(cfg.dataset)
        testA_loader = build_loader(cfg, testA_dataset, dist.get_rank(), shuffle=False)

        cfg.dataset.split = "testB"
        testB_dataset = instantiate(cfg.dataset)
        testB_loader = build_loader(cfg, testB_dataset, dist.get_rank(), shuffle=False)
        prefixs.extend(['testA','testB'])
        loaders.extend([testA_loader,testB_loader])
    else:
        cfg.dataset.split = "test"
        test_dataset=instantiate(cfg.dataset)
        test_loader=build_loader(cfg, test_dataset, dist.get_rank(), shuffle=False)
        prefixs.append('test')
        loaders.append(test_loader)
    
    # build model
    cfg.model.language_encoder.pretrained_emb = train_set.pretrained_emb
    cfg.model.language_encoder.token_size = train_set.token_size
    model = instantiate(cfg.model)

    # build optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)


    torch.cuda.set_device(dist.get_rank())
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)
    model_without_ddp = model.module

    if is_main_process():
        total_params = sum([param.nelement() for param in model.parameters()])
        trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logger.info(str(model))
        logger.info("Number of all params: %.2fM" % (total_params / 1e6))
        logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))


    checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage.cuda() )
    model_without_ddp.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if cfg.train.amp:
        assert torch.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    if is_main_process():
        writer = SummaryWriter(log_dir=cfg.train.output_dir)
    else:
        writer = None

    save_ids=np.random.randint(1, len(val_loader) * cfg.train.batch_size, 100) if cfg.train.log_image else None
    for data_loader, prefix in zip(loaders, prefixs):
        box_ap, mask_ap = validate(
            cfg=cfg, 
            model=model, 
            data_loader=data_loader, 
            writer=writer, 
            epoch=0, 
            ix_to_token=val_set.ix_to_token,
            logger=logger,
            rank=dist.get_rank(),
            save_ids=save_ids,
            prefix=prefix)
        logger.info(f' * BoxIoU@0.5 {box_ap:.3f} MaskIoU {mask_ap:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimREC")
    parser.add_argument('--config', type=str, required=True, default='./config/simrec_refcoco_scratch.py')
    parser.add_argument('--eval-weights', type=str, required=True, default='')
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
    logger = create_logger(output_dir=cfg.train.output_dir, dist_rank=dist.get_rank())

    # Refine cfg for evaluation
    cfg.train.resume_path = args.eval_weights
    logger.info(f"Running evaluation from specific checkpoint {cfg.train.resume_path}......")

    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "config.yaml")
        LazyConfig.save(cfg, path)

    main(cfg)
