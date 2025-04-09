import argparse
import datetime
import numpy as np
import os
import time
import math
import sys
from pathlib import Path

import torch
import torch.utils.data
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from copy import deepcopy
# from torch.utils.tensorboard import SummaryWriter
import logging
from torch import nn
from torch.nn.parallel import DistributedDataParallel
#assert timm.__version__ == "0.3.2"  # version check
# import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.logger import create_logger, AverageMeter, ProgressMeter, MaxMeter

from util.misc import NativeScalerWithGradNormCount as NativeScaler

from util import add_weight_decay, cosine_scheduler, nested_to_gpu

from data_utils import build_pretrain_dataset, get_loader
from models import init_jepa_model
from masks import build_mask_collator

from opts import parser



def main(args):
    misc.init_dist_pytorch(args)
    # args.rank = 0
    create_logger(args)
    logging.info(f'job dir: {args.output_dir}')
    logging.info("{}".format(str(args)).replace(', ', ',\n'))
    device = torch.device(args.gpu)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_pretrain_dataset(args, args.ssl_type)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset_train,
        num_replicas=args.world_size,
        rank=args.rank)
    mask_collator = build_mask_collator(args)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        collate_fn=mask_collator,
        sampler=train_sampler,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True)
    
    model = init_jepa_model(
        device=device,
        args=args,
        ssl_type=args.ssl_type
    )
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logging.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    logging.info("actual lr: %.2e" % args.lr)

    logging.info("accumulate grad iterations: %d" % args.accum_iter)
    logging.info("effective batch size: %d" % eff_batch_size)

    model = DistributedDataParallel(model, static_graph=True)
    model_without_ddp = model.module
    # model = torch.compile(model)
    # encoder = DistributedDataParallel(encoder, static_graph=True)
    # predictor = DistributedDataParallel(predictor, static_graph=True)
    # target_encoder = DistributedDataParallel(target_encoder)
    
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=args.betas)
    
    iters_per_epoch = int(args.ipe_scale * len(data_loader_train))
    lr_schedule = cosine_scheduler(
        args.lr,  # linear scaling rule
        args.min_lr,
        args.epochs, iters_per_epoch,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.start_lr
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, iters_per_epoch,
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(args.ema, args.ema_end, args.epochs, iters_per_epoch)
    if args.amp:
        loss_scaler = NativeScaler()
    else:
        raise NotImplementedError
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, new_start=args.new_start)
    
    
    start_time = time.time()
    stop_epoch = args.stop_epoch if args.stop_epoch else args.epochs
    for epoch in range(args.start_epoch, stop_epoch):
        epoch_start = time.time()
        data_loader_train.sampler.set_epoch(epoch)
        # data_time = AverageMeter('Data Time', ":6.3f")
        batch_time = AverageMeter('Time', ':6.3f')
        loss_meter = AverageMeter(f'Loss', ':.4f')
        loss_intra_meter = AverageMeter(f'Loss_Intra', ':.4f')
        loss_extra_meter = AverageMeter(f'Loss_Extra', ':.4f')
        maskA_meter = AverageMeter(f'Mask_Enc', ':.1f')
        maskB_meter = AverageMeter(f'Mask_Pred', ':.1f')
        norm_meter = AverageMeter(f'Grad_Norm', ':.2f')
        var_meter = AverageMeter('Pred_Var', ':.2f')
        meters = [batch_time, loss_meter, loss_intra_meter, loss_extra_meter, maskA_meter, maskB_meter, norm_meter, var_meter]
        progress = ProgressMeter(
            len(data_loader_train),
            meters,
            prefix=f"[Epoch {epoch}] ")
        end = time.time()
        # for data_iter_step, (imgs, masks_enc, masks_pred) in enumerate(data_loader_train):
        for data_iter_step, data in enumerate(data_loader_train):
            it = len(data_loader_train) * epoch + data_iter_step
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if param_group['weight_decay'] > 0:
                    param_group["weight_decay"] = wd_schedule[it]
            data = nested_to_gpu(data, device)
            # imgs = imgs.to(device, non_blocking=True)
            # masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            # masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]
            masks_enc, masks_pred = data[-2], data[-1]

            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # loss = model(imgs, masks_enc, masks_pred, args.loss_type, args.target_last_k, args.target_norm_type, args.target_type)
                outputs = model(data, args)
                loss = outputs['loss']
                pred_var = outputs['pred_var']
            loss_value = loss.item()
            loss = loss / args.accum_iter
            update_grad = ((it+1) % args.accum_iter == 0)
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=update_grad)
            if update_grad:
                # optimizer.zero_grad()
                optimizer.zero_grad(set_to_none=True)
                if args.pretrained == '':
                    model.module.update_target_encoder(momentum_schedule[it])
                norm_meter.update(grad_norm)
            batch_time.update(time.time() - end)
            loss_meter.update(loss_value)
            if 'loss_intra' in outputs:
                loss_intra_meter.update(outputs['loss_intra'])
                loss_extra_meter.update(outputs['loss_extra'])
            var_meter.update(pred_var)
            torch.cuda.synchronize()
            if (data_iter_step+1) % args.print_freq == 0 or data_iter_step == len(data_loader_train) - 1:
                logging.info(progress.display(data_iter_step) + f'\tLR={lr_schedule[it]:.4e}\tWD={wd_schedule[it]:.4e}\tEMA={momentum_schedule[it]:.4e}')
            end = time.time()
        
        if args.rank == 0:
            if (epoch+1) % args.eval_freq == 0 or epoch == args.epochs - 1 or epoch in args.eval_list:
                to_save = {
                            'model': model_without_ddp.state_dict(),
                            'epoch': epoch,
                            'args': args
                            }
                torch.save(to_save, os.path.join(args.output_dir, f'epoch_{epoch}.pth.tar'))
            misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch, is_best=False)
        epoch_time = time.time() - epoch_start
        logging.info(f'Epoch time {str(datetime.timedelta(seconds=int(epoch_time)))}')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))
    if args.rank == 0 and stop_epoch == args.epochs:
        os.remove(os.path.join(args.output_dir, 'checkpoint.pth'))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.exp_name:
        args.output_dir = os.path.join(args.output_dir, args.exp_name, f'seed{args.seed}')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
