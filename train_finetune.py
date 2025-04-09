import argparse
import datetime
import json
import numpy as np
import os
import time
import math
import sys
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.utils.data
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from copy import deepcopy
# from torch.utils.tensorboard import SummaryWriter
import logging

#assert timm.__version__ == "0.3.2"  # version check
# import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.logger import create_logger, AverageMeter, ProgressMeter, MaxMeter

from util.misc import NativeScalerWithGradNormCount as NativeScaler

from util import build_optimizer, param_groups_lrd

from util.metrics import cal_metrics
from sklearn.metrics import accuracy_score

from data_utils.xray_transform import build_transform_xray
from data_utils import get_loader, get_infinite_loader, build_dataset
from data_utils.chest_dset import MedFMC_Chest

from models import build_transfer_model
from opts import parser


@torch.no_grad()
def validate(model, data_loader, criterion, args=None, test=False, ten_crop=False):

    loss_meter = AverageMeter(f'Loss', ':.4f')
    # switch to evaluation mode
    model.eval()
    end  = time.time()
    all_output = []
    all_label = []
    for i, batch in enumerate(data_loader):
        images = batch[0]
        target = batch[-1].float()
        all_label.append(target)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            if ten_crop:
                bs, ncrops, c, h, w = images.size()
                outputs = model(images.view(-1, c, h, w)).view(bs, ncrops, -1).mean(1)
            else:
                outputs = model(images)
            loss = criterion(outputs, target)
        _b = images.shape[0]

        all_output.append(outputs.cpu())
        loss_meter.update(loss.item(), _b)

    sign = 'TEST' if test else 'VAL'
    logging.info(f'{sign} on {len(data_loader.dataset)} images')
    if ten_crop:
        sign = sign + '_TEN'
    all_label = torch.cat(all_label, dim=0).numpy()
    outputs = torch.cat(all_output, dim=0).numpy()
    if args.dataset == 'rsna_3cls':
        acc = accuracy_score(np.argmax(all_label,axis=1),np.argmax(outputs,axis=1))
        logging.info(f'[{sign}] Acc={acc:.4f}')
        return acc, loss_meter.avg
    else:
        res = cal_metrics(outputs, all_label)
        res['loss'] = loss_meter.avg
        logging.info(f"[{sign}] mAUC={res['mauc']:.4f}, mAUPR={res['maupr']:.4f}, loss={res['loss']:.4f}")
        auc_str = [f"{m:.4f}" for m in res['cls_auc']]
        logging.info(f'Class AUROC = {auc_str}')
        aupr_str = [f"{m:.4f}" for m in res['cls_aupr']]
        logging.info(f'Class AUPR = {aupr_str}')
        return res['mauc'], loss_meter.avg


def train_iter(model, optimizer, criterion, loss_scaler, samples, targets):
    model.train(True)
    samples = samples.cuda(non_blocking=True)
    targets = targets.float().cuda(non_blocking=True)
    _b = samples.shape[0]


    with torch.cuda.amp.autocast():
        outputs = model(samples)
        loss = criterion(outputs, targets)

    loss_value = loss.item()

    if not math.isfinite(loss_value):
        print("Loss is {}, stopping training".format(loss_value))
        sys.exit(1)

    # update_grad = (data_iter_step + 1) % accum_iter == 0
    update_grad = True
    if args.amp:
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=update_grad)
    else:
        loss.backward()
        if update_grad:
            if args.clip_grad is not None:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
    if update_grad:
        optimizer.zero_grad()
    # torch.cuda.synchronize()
    return loss


def main(args):
    misc.init_dist_pytorch(args)
    # args.rank = 0
    create_logger(args)
    logging.info(f'job dir: {args.output_dir}')
    logging.info("{}".format(str(args)).replace(', ', ',\n'))
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(args, split='train')
    dataset_val = build_dataset(args, split='val')
    dataset_test = build_dataset(args, split='test', ten_crop=args.ten_crop)

    # dataset_test = build_dataset(args, split='test')
    
    # data_loader_train = get_infinite_loader(args, dataset_train, batch_size=args.batch_size)
    data_loader_train = get_loader(args, dataset_train, is_train=True)
    data_loader_val = get_loader(args, dataset_val, is_train=False, custom_bs=args.batch_size_eval)
    data_loader_test = get_loader(args, dataset_test, is_train=False, custom_bs=args.batch_size // 4)
    # data_loader_test = get_loader(args, dataset_test, is_train=False, custom_bs=args.batch_size_eval)
    args.num_classes = dataset_train.num_classes
    
    model = build_transfer_model(args)
    # freeze_model(args, model)
    
    model.to(device)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logging.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    logging.info("actual lr: %.2e" % args.lr)

    logging.info("accumulate grad iterations: %d" % args.accum_iter)
    logging.info("effective batch size: %d" % eff_batch_size)

    if args.dataset == 'rsna_3cls':
        criterion = nn.CrossEntropyLoss()
    else:
        pos_weight = torch.Tensor([args.bce_pos_weight]*args.num_classes) if args.bce_pos_weight is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()
    logging.info(f'criterion: {criterion}')

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    model_without_ddp = model.module
    param_groups = param_groups_lrd(model_without_ddp, args.weight_decay, no_weight_decay_list=['pos_embed', 'cls_token', 'head.cls_query', 'prompt'], layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=args.betas)
    logging.info(f'Optimizer: LR={args.lr}, betas={args.betas}')
    if args.amp:
        loss_scaler = NativeScaler()
    else:
        raise NotImplementedError
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, new_start=args.new_start)
    
    if args.eval:
        # validate(model, data_loader_val, criterion, args=args)
        validate(model, data_loader_test, criterion, args=args, test=True, ten_crop=args.ten_crop)
        return
    start_time = time.time()
    

    
    # best_state = None
    tolerence = 0
    if args.dataset == 'rsna_3cls':
        val_meter = MaxMeter('val acc')
    else:
        val_meter = MaxMeter('val auroc')
    val_loss_meter = MaxMeter('val loss', mode='min')
    for epoch in range(args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        # data_time = AverageMeter('Data Time', ":6.3f")
        batch_time = AverageMeter('Time', ':6.3f')
        loss_meter = AverageMeter(f'Loss', ':.4f')
        meters = [batch_time, loss_meter]
        progress = ProgressMeter(
            len(data_loader_train),
            meters,
            prefix=f"[Epoch {epoch}] ")
        end = time.time()
        model.train(True)
        for data_iter_step, (samples, targets) in enumerate(data_loader_train):
            # data_time.update(time.time() - end)
            lr = adjust_learning_rate(optimizer, epoch + data_iter_step/len(data_loader_train), args)
            samples = samples.cuda(non_blocking=True)
            targets = targets.float().cuda(non_blocking=True)
            _b = samples.shape[0]
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
            loss_value = loss.item()
            loss = loss / args.accum_iter
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            update_grad = (data_iter_step + 1) % args.accum_iter == 0
            if args.amp:
                loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                            parameters=model.parameters(), create_graph=False,
                            update_grad=update_grad)
            else:
                loss.backward()
                if update_grad:
                    if args.clip_grad is not None:
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            if update_grad:
                optimizer.zero_grad(set_to_none=True)
            # torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            loss_meter.update(loss.item(), args.batch_size)
        
            if data_iter_step % args.print_freq == 0:
                logging.info(progress.display(data_iter_step) + f'\tLR = {lr:.4e}')
            end = time.time()
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_stat, val_loss = validate(model, data_loader_val, criterion, args=args)
            is_best = val_meter.update(val_stat, epoch)
            loss_is_best = val_loss_meter.update(val_loss, epoch)
            is_best = is_best or loss_is_best
            # val_aupr_meter.update(val_stat['maupr'], epoch)
            logging.info(val_meter.display())
            logging.info(val_loss_meter.display())

            # logging.info(val_aupr_meter.display())
            if is_best:
                # best_state = deepcopy(model.state_dict())
                tolerence = 0
                test_stat, test_loss = validate(model, data_loader_test, criterion, args=args, ten_crop=args.ten_crop, test=True)
                logging.info(f'Test At Best: {test_stat:.4f}')
            else:
                tolerence += 1
           
            to_save = {
                        'model': model_without_ddp.state_dict(),
                        'epoch': epoch,
                        'args': args
                        }
            if args.save_mode == 'best' and args.rank == 0 and is_best:
                torch.save(to_save, os.path.join(args.output_dir, f'model_best.pth.tar'))
            if args.save_mode == 'every' and args.rank == 0:
                torch.save(to_save, os.path.join(args.output_dir, f'epoch_{epoch}.pth.tar'))

        if args.early_stop is not None and tolerence >= args.early_stop:
            logging.info('Early Stop!')
            break
        end = time.time()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))
    # if args.rank == 0:
    #     os.remove(os.path.join(args.output_dir, 'checkpoint.pth'))
    # logging.info('=== Testing ===')
    # model.load_state_dict(best_state)
    # test_stat = validate(model, data_loader_test, criterion, args=args, test=True)
    # logging.info('=== Complete ===')


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    args = parser.parse_args()
    train_params = f'dp{args.drop_path}_ld{args.layer_decay}_lr{args.lr}_ep{args.epochs}w{args.warmup_epochs}_cat_{args.dataset_cat}_bs{args.batch_size}_wd{args.weight_decay}_aug{args.aug_type}_jit{args.color_jitter}_crop{args.crop_type}_s{args.scale_min}'
    args.output_dir = os.path.join(args.output_dir, args.exp_name, train_params, f'seed{args.seed}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
