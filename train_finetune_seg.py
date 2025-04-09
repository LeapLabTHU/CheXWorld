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
from monai.losses.dice import DiceCELoss, DiceLoss
from copy import deepcopy
# from torch.utils.tensorboard import SummaryWriter
import logging

#assert timm.__version__ == "0.3.2"  # version check
# import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.logger import create_logger, AverageMeter, ProgressMeter, MaxMeter

from util.misc import NativeScalerWithGradNormCount as NativeScaler

from util import build_optimizer, param_groups_lrd, add_weight_decay

from util.metrics import cal_metrics

from data_utils.xray_transform import build_transform_xray
from data_utils import get_loader, get_infinite_loader,  build_seg_dataset
from data_utils.chest_dset import MedFMC_Chest

from models import build_transfer_model
from opts import parser


def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(bool)
    im2 = np.asarray(im2 > 0.5).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def mean_dice_coef(y_true,y_pred):
    sum=0
    for i in range (y_true.shape[0]):
        sum += dice(y_true[i,:,:,:],y_pred[i,:,:,:])
    return sum/y_true.shape[0]




@torch.no_grad()
def validate(model, data_loader, criterion, args=None, test=False, ten_crop=False, thres=0.5):

    loss_meter = AverageMeter(f'Loss', ':.4f')
    dice_meter = AverageMeter(f'Dice', ':.4f')
    dice_pos_meter = AverageMeter(f'Dice_Pos', ':.4f')
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
            output = model(images) # [B,C,H,W]
            loss = criterion(output, target)
            output = F.sigmoid(output)
            # output[output>thres] =1
            # output[output<=thres] = 0
            output = (output > thres).float()
        
        # post proc
        if args.seg_zero_thres is not None:
            mask_cnt = output.sum([-3, -2, -1])
            invalid = mask_cnt < args.seg_zero_thres
            output[invalid] = 0
        mdice = mean_dice_coef(output.cpu().numpy(), target.cpu().numpy())
        _b = images.shape[0]
        dice_meter.update(mdice, _b)
        loss_meter.update(loss.item(), _b)
        target_cnt = target.sum([-3, -2, -1])
        positive = target_cnt > 0
        if positive.sum() > 0:
            mdice_pos = mean_dice_coef(output[positive].cpu().numpy(), target[positive].cpu().numpy())
            dice_pos_meter.update(mdice_pos, positive.sum().item())

    logging.info(f'Val Loss: {loss_meter.avg}')
    logging.info(f'Avg dice for each case ({dice_meter.count}): {dice_meter.avg} (thres={thres:.2f}, zero_thres={args.seg_zero_thres})')
    logging.info(f'Dice Positive ({dice_pos_meter.count}): {dice_pos_meter.avg:.4f} (thres={thres:.2f}, zero_thres={args.seg_zero_thres})')
    return dice_meter.avg

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
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    dataset_train = build_seg_dataset(args, split='train')
    dataset_val = build_seg_dataset(args, split='val')
    dataset_test = build_seg_dataset(args, split='test')

    # dataset_test = build_dataset(args, split='test')
    
    # data_loader_train = get_infinite_loader(args, dataset_train, batch_size=args.batch_size)
    data_loader_train = get_loader(args, dataset_train, is_train=True)
    data_loader_val = get_loader(args, dataset_val, is_train=False, custom_bs=args.batch_size_eval)
    data_loader_test = get_loader(args, dataset_test, is_train=False, custom_bs=args.batch_size_eval)
    
    # data_loader_test = get_loader(args, dataset_test, is_train=False, custom_bs=args.batch_size_eval)
    args.num_classes = dataset_train.num_classes
    
    model = build_transfer_model(args, seg=True)
    # freeze_model(args, model)
    
    model.to(device)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logging.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    logging.info("actual lr: %.2e" % args.lr)

    logging.info("accumulate grad iterations: %d" % args.accum_iter)
    logging.info("effective batch size: %d" % eff_batch_size)

    criterion = DiceLoss(include_background=False, sigmoid=True)
    criterion_ce = torch.nn.BCEWithLogitsLoss()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module
    if 'vit' not in args.model:
        param_groups = add_weight_decay(model_without_ddp, weight_decay=args.weight_decay)
    else:
        backbone_param_groups = param_groups_lrd(model_without_ddp.feature_model, args.weight_decay, no_weight_decay_list=['pos_embed', 'cls_token', 'head.cls_query', 'prompt'], layer_decay=args.layer_decay)

        if args.backbone_lr_scale < 1.0:
            logging.info(f'Backbone lr scale = {args.backbone_lr_scale}')
            for group in backbone_param_groups:
                group['lr_scale'] = group['lr_scale'] * args.backbone_lr_scale
        head_param_groups = add_weight_decay(model_without_ddp.head_parameters(), weight_decay=args.weight_decay)
        if args.backbone_freeze:
            for p in model_without_ddp.feature_model.parameters():
                p.requires_grad = False
            param_groups = head_param_groups
        else:
            param_groups = backbone_param_groups + head_param_groups
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    
    if args.amp:
        loss_scaler = NativeScaler()
    else:
        raise NotImplementedError
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, new_start=args.new_start)
    
    if args.eval:
        validate(model, data_loader_test, criterion, args=args, thres=args.seg_thres)
        # validate(model, data_loader_test, criterion, args=args, test=True)
        return
    start_time = time.time()
    

    
    # best_state = None
    tolerence = 0
    val_dice_meter = MaxMeter('val dice')
    
    for epoch in range(args.start_epoch, args.epochs):
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
                loss = criterion(outputs, targets) + criterion_ce(outputs, targets)
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

            val_stat = validate(model, data_loader_val, criterion, args=args, thres=args.seg_thres)
            is_best = val_dice_meter.update(val_stat, epoch)
            logging.info(val_dice_meter.display())
            if is_best:
                # best_state = deepcopy(model.state_dict())
                tolerence = 0
                test_stat = validate(model, data_loader_test, criterion, args=args, test=True)
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
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, is_best=False)
        if args.early_stop is not None and tolerence >= args.early_stop:
            logging.info('Early Stop!')
            break
        end = time.time()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))
    if args.rank == 0:
        os.remove(os.path.join(args.output_dir, 'checkpoint.pth'))
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
    train_params = f'dp{args.drop_path}_ld{args.layer_decay}_lr{args.lr}_ep{args.epochs}w{args.warmup_epochs}_bs{args.batch_size}_wd{args.weight_decay}_aug{args.aug_type}_jit{args.color_jitter}_crop{args.crop_type}_s{args.scale_min}'
    args.output_dir = os.path.join(args.output_dir, args.exp_name, train_params, f'seed{args.seed}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
