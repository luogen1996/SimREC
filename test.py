import os
import time
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

from simrec.config import instantiate, LazyConfig


from simrec.datasets.dataloader import build_loader
from simrec.datasets.utils import yolobox2label
from simrec.models.utils import batch_box_iou, mask_processing, mask_iou
from simrec.utils.distributed import seed_everything, main_process, reduce_meters, find_free_port
from simrec.utils.visualize import draw_visualization, normed2original
from simrec.utils.env import seed_everything, setup_unique_version
from simrec.utils.metric import AverageMeter, ProgressMeter


def validate(cfg,
             net,
             loader,
             writer,
             epoch,
             rank,
             ix_to_token,
             save_ids=None,
             prefix='Val',
             ema=None):
    if ema is not None:
        ema.apply_shadow()
    net.eval()

    batches = len(loader)
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
    progress = ProgressMeter(cfg.train.version, cfg.train.epochs, len(loader), meters, prefix=prefix+': ')
    with torch.no_grad():
        end = time.time()
        for ith_batch, data in enumerate(loader):
            ref_iter, image_iter, mask_iter, box_iter,gt_box_iter, mask_id, info_iter = data
            ref_iter = ref_iter.cuda( non_blocking=True)
            image_iter = image_iter.cuda( non_blocking=True)
            box_iter = box_iter.cuda( non_blocking=True)
            box, mask= net(image_iter, ref_iter)


            gt_box_iter=gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])
            gt_box_iter=gt_box_iter.cpu().numpy()
            info_iter=info_iter.cpu().numpy()
            box=box.squeeze(1).cpu().numpy()
            pred_box_vis=box.copy()

            ###predictions to gt
            for i in range(len(gt_box_iter)):
                box[i]=yolobox2label(box[i],info_iter[i])


            box_iou=batch_box_iou(torch.from_numpy(gt_box_iter),torch.from_numpy(box)).cpu().numpy()
            seg_iou=[]
            mask=mask.cpu().numpy()
            for i, mask_pred in enumerate(mask):
                if writer is not None and save_ids is not None and ith_batch*cfg.train.BATCH_SIZE+i in save_ids:
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
                    det_image=draw_visualization(normed2original(image_iter[i], cfg.dataset.transforms[1].mean, cfg.dataset.transforms[1].std),sent,pred_box_vis[i].cpu().numpy(),box_iter[i].cpu().numpy())
                    writer.add_image('image/' + str(ith_batch * cfg.train.batch_size + i) + '_det',det_image,epoch,dataformats='HWC')
                    writer.add_image('image/' + str(ith_batch * cfg.train.batch_size + i) + '_seg', (mask[i,None]*255).astype(np.uint8))

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


                mask_gt=np.load(os.path.join(cfg.dataset.mask_path[cfg.dataset.dataset],'%d.npy'%mask_id[i]))
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

            if (ith_batch % cfg.train.print_freq == 0 or ith_batch==(len(loader)-1)) and main_process(cfg, rank):
                progress.display(epoch, ith_batch)
            batch_time.update(time.time() - end)
            end = time.time()

        if main_process(cfg, rank) and writer is not None:
            writer.add_scalar("Acc/BoxIoU@0.5", box_ap.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/MaskIoU", mask_ap.avg_reduce, global_step=epoch)
            writer.add_scalar("Acc/IE", inconsistency_error.avg_reduce, global_step=epoch)
            for item in mask_aps:
                writer.add_scalar("Acc/MaskIoU@%.2f"%item, np.array(mask_aps[item]).mean(), global_step=epoch)
    if ema is not None:
        ema.restore()
    return box_ap.avg_reduce, mask_ap.avg_reduce


def main_worker(gpu, cfg):
    global best_det_acc,best_seg_acc
    best_det_acc,best_seg_acc=0.,0.
    
    if cfg.train.distributed.enabled:
        if cfg.train.distributed.dist_url == "env://" and cfg.train.distributed.rank == -1:
            cfg.train.distributed.rank = int(os.environ["RANK"])
        if cfg.train.distributed.enabled:
            cfg.train.distributed.rank = cfg.train.distributed.rank * len(cfg.train.distributed.gpus) + gpu
        dist.init_process_group(
            backend=dist.Backend('NCCL'), 
            init_method=cfg.train.distributed.dist_url, 
            world_size=cfg.train.distributed.world_size, 
            rank=cfg.train.distributed.rank
        )

    # build training dataset and dataloader
    cfg.dataset.split = "train"
    train_set = instantiate(cfg.dataset)
    train_loader=build_loader(
        cfg,
        train_set, 
        gpu,
        shuffle=(not cfg.train.distributed.enabled)
    )

    # build single or multi-datasets for validation
    loaders=[]
    prefixs=['val']
    cfg.dataset.split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader=build_loader(
        cfg,
        val_set,
        gpu,
        shuffle=False
    )
    loaders.append(val_loader)
    
    if cfg.dataset.dataset == 'refcoco' or cfg.dataset.dataset == 'refcoco+':
        cfg.dataset.split = "testA"
        testA_dataset = instantiate(cfg.dataset)
        testA_loader = build_loader(cfg, testA_dataset, gpu, shuffle=False)

        cfg.dataset.split = "testB"
        testB_dataset = instantiate(cfg.dataset)
        testB_loader = instantiate(cfg, testB_dataset, gpu, shuffle=False)
        prefixs.extend(['testA','testB'])
        loaders.extend([testA_loader,testB_loader])
    else:
        cfg.dataset.split = "test"
        test_dataset=instantiate(cfg.dataset)
        test_loader=build_loader(cfg, test_dataset, gpu, shuffle=False)
        prefixs.append('test')
        loaders.append(test_loader)
    
    # build model
    cfg.model.language_encoder.pretrained_emb = train_set.pretrained_emb
    cfg.model.language_encoder.token_size = train_set.token_size
    net = instantiate(cfg.model)

    # build optimizer
    params = filter(lambda p: p.requires_grad, net.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)


    if cfg.train.distributed.enabled:
        torch.cuda.set_device(gpu)
        net = DDP(net.cuda(), device_ids=[gpu],find_unused_parameters=True)
    elif len(gpu)==1:
        net.cuda()
    else:
        net = DP(net.cuda())

    if main_process(cfg, gpu):
        print(cfg)
        total = sum([param.nelement() for param in net.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))
        total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6))


    if os.path.isfile(cfg.train.resume_path):
        checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage.cuda() )
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        if main_process(cfg, gpu):
            print("==> loaded checkpoint from {}\n".format(cfg.train.resume_path) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))

    if cfg.train.amp:
        assert torch.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    if main_process(cfg, gpu):
        writer = SummaryWriter(log_dir=os.path.join(cfg.train.log_path, str(cfg.train.version)))
    else:
        writer = None

    save_ids=np.random.randint(1, len(val_loader) * cfg.train.batch_size, 100) if cfg.train.log_image else None
    for loader_,prefix_ in zip(loaders,prefixs):
        print()
        box_ap,mask_ap=validate(cfg, net, loader_ , writer, 0, gpu, val_set.ix_to_token,save_ids=save_ids,prefix=prefix_)
        print(box_ap, mask_ap)


def main():
    parser = argparse.ArgumentParser(description="SimREC")
    parser.add_argument('--config', type=str, required=True, default='./config/simrec_refcoco_scratch.py')
    parser.add_argument('--eval-weights', type=str, required=True, default='')
    args=parser.parse_args()
    cfg = LazyConfig.load(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.train.gpus)
    setup_unique_version(cfg)
    seed_everything(cfg.train.seed)
    N_GPU=len(cfg.train.gpus)
    cfg.train.resume_path=args.eval_weights
    if not os.path.exists(os.path.join(cfg.train.log_path,str(cfg.train.version))):
        os.makedirs(os.path.join(cfg.train.log_path, str(cfg.train.version),'ckpt'),exist_ok=True)

    if N_GPU == 1:
        cfg.train.distributed.enabled = False
    else:
        # turn on single or multi node multi gpus training
        cfg.train.distributed.enabled = True
        cfg.train.distributed.world_size *= N_GPU
        cfg.train.distributed.dist_url = f"tcp://127.0.0.1:{find_free_port()}"
    if cfg.train.distributed.enabled:
        mp.spawn(main_worker, args=(cfg,), nprocs=N_GPU, join=True)
    else:
        main_worker(cfg.train.gpus, cfg)


if __name__ == '__main__':
    main()
