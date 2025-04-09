import argparse

parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

## exp config
parser.add_argument('--dataset', default='mimic', type=str)
parser.add_argument('--data_path', default='', type=str,
                    help='dataset path')
parser.add_argument('--dataset_cat', type=int, default=1)
parser.add_argument('--dataset_balance', type=int, default=0)
parser.add_argument('--dataset_fold', type=int, default=0)
parser.add_argument('--dataset_seed', type=int, default=42)

parser.add_argument('--fmc_shots', type=int, default=10)
parser.add_argument('--fmc_exp_num', type=int, default=1)

parser.add_argument('--output_dir', default='../prompt_output',
                    help='path where to save, empty for no saving')
parser.add_argument('--exp_name', default='', type=str)
parser.add_argument('--print_freq', default=20, type=int)
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--eval_list', type=int, nargs='*', default=[])
parser.add_argument('--save_mode', type=str, default='best', choices=['best', 'every', 'none'])
parser.add_argument('--seed', type=int, default=2048)
parser.add_argument('--data_seed', type=int, default=None)

# Model parameters
parser.add_argument('--model', default='vit_base', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--ssl_type', default='jepa', type=str)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--input_size', default=224, type=int,
                    help='images input size')
parser.add_argument('--resize_size', default=None, type=int,
                    help='images resize size')
parser.add_argument('--mask_ratio', default=0.75, type=float,
                    help='Masking ratio (percentage of removed patches).')

parser.add_argument('--norm_pix_loss', action='store_true',
                    help='Use (per-patch) normalized pixels as targets for computing loss')
parser.set_defaults(norm_pix_loss=False)

# Optimizer parameters
parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999))
parser.add_argument('--optim', type=str, default='adamw',
                    help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='sgd momentum')
parser.add_argument('--weight_decay', type=float, default=0.04,
                    help='weight decay')
parser.add_argument('--weight_decay_end', type=float, default=0.4,
                    help='weight decay')
parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (absolute lr)')
parser.add_argument('--start_lr', type=float, default=0.0)
parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--epochs', default=400, type=int)
parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                    help='epochs to warmup LR')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
parser.add_argument('--batch_size_eval', default=256, type=int,
                    help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

parser.add_argument('--accum_iter', default=1, type=int,
                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
parser.add_argument('--clip_grad', type=float, default=None)
parser.add_argument('--amp', action='store_true', default=False)
parser.add_argument('--dist_backend', type=str, default='nccl')

# Dataset parameters
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')
parser.add_argument('--new_start', action='store_true', default=False)

parser.add_argument('--start_epoch', default=0, type=int,help='start epoch')
parser.add_argument('--stop_epoch', default=None, type=int)
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
parser.set_defaults(pin_mem=True)

parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--const_lr', action='store_true', default=False)

### finetune
parser.add_argument('--finetune', default='',
                    help='finetune from checkpoint')
parser.add_argument('--linprobe', action='store_true', default=False)


parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                    help='Drop path rate (default: 0.0)')
parser.add_argument('--drop_path_uniform', action='store_true', default=False)
parser.add_argument('--drop', type=float, default=0., metavar='PCT',
                    help='Drop Out rate (default: 0.0)')
parser.add_argument('--layer_decay', type=float, default=1.0,
                    help='layer-wise lr decay from ELECTRA/BEiT')
parser.add_argument('--color_jitter', type=float, default=0.2, metavar='PCT',
                    help='Color jitter factor (enabled only when not using Auto/RandAug)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),


# parser.add_argument('--timm_aug', action='store_true', default=False)
# parser.add_argument('--affine_aug', action='store_true', default=False)
parser.add_argument('--crop_type', type=str, default='rc', choices=['rc', 'rrc'])
parser.add_argument('--aug_type', type=str, default='aff')
parser.add_argument('--scale_min', type=float, default=0.08)

parser.add_argument('--ra_n', type=int, default=2)
parser.add_argument('--ra_m', type=int, default=7)
parser.add_argument('--rot', type=float, default=10)
# * Random Erase params
# parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
#                     help='Random erase prob (default: 0.25)')
# parser.add_argument('--remode', type=str, default='pixel',
#                     help='Random erase mode (default: "pixel")')
# parser.add_argument('--recount', type=int, default=1,
#                     help='Random erase count (default: 1)')
# parser.add_argument('--resplit', action='store_true', default=False,
#                     help='Do not random erase first (clean) augmentation split')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--smoothing', type=float, default=0,
                    help='Label smoothing (default: 0)')

# for xray normalize
parser.add_argument('--norm_type', type=str, default='default')
parser.add_argument('--data_pct', type=float, default=1.0)


parser.add_argument('--bce_pos_weight', type=int, default=None)

# grad ckpt
parser.add_argument('--grad_ckpt', action='store_true', default=False)

parser.add_argument('--global_pool', type=str, default='token')
parser.add_argument('--pretrained', type=str, default='')
parser.add_argument('--with_cls_token', action='store_true', default=False)

parser.add_argument('--early_stop', type=int, default=None)
## train mode
parser.add_argument('--freeze_mode', type=str, default='')

parser.add_argument('--mask_type', type=str, default='multiblock')
parser.add_argument('--loss_type', type=str, default='l1')
parser.add_argument('--target_last_k', type=int, default=1)
parser.add_argument('--target_norm_type', type=str, default='avg_ln')
parser.add_argument('--target_type', type=str, default='') # normal input

parser.add_argument('--pred_depth', type=int, default=6)
parser.add_argument('--pred_emb_dim', type=int, default=384)
parser.add_argument('--ema', type=float, default=0.996)
parser.add_argument('--ema_end', type=float, default=1.0)
parser.add_argument('--use_target', action='store_true', default=False)

parser.add_argument('--mae_mask_ratio', type=float, default=0.75)
parser.add_argument('--ipe_scale', type=float, default=1.0) # from I-JEPA

parser.add_argument('--mask_nenc', type=int, default=1)
parser.add_argument('--mask_npred', type=int, default=4)
parser.add_argument('--mask_min_keep', type=int, default=10)
parser.add_argument('--mask_max_keep', type=int, default=None)
parser.add_argument('--mask_rand_keep', action='store_true', default=False)
parser.add_argument('--mask_merge', action='store_true', default=False)

parser.add_argument('--enc_mask_scale', type=float, nargs=2, default=(0.85, 1.0))
parser.add_argument('--pred_mask_scale', type=float, nargs=2, default=(0.15, 0.2))
parser.add_argument('--extra_global_scale', type=float, nargs=2, default=(0.3, 1.0))
parser.add_argument('--extra_local_scale', type=float, nargs=2, default=(0.05, 0.3))
parser.add_argument('--extra_num_local', type=int, default=1)
parser.add_argument('--extra_mean', action='store_true', default=False)
parser.add_argument('--extra_shuffle_mask', action='store_true', default=False)
parser.add_argument('--extra_loss_weight', type=float, default=1.0)

parser.add_argument('--preload_dataset', action='store_true', default=False)

parser.add_argument('--sscale', action='store_true', default=False)

parser.add_argument('--data_postfix', type=str, default='')
parser.add_argument('--no_stdout', action='store_true', default=False) 
parser.add_argument('--iwm_disable', action='store_true', default=False)
parser.add_argument('--rel_pos_disable', action='store_true', default=False)
parser.add_argument('--pos_disable', action='store_true', default=False)
## iwm aug params
parser.add_argument('--iwm_jitter_prob', type=float, default=0.8)
parser.add_argument('--iwm_blur_prob', type=float, default=0.2)
parser.add_argument('--iwm_noise_prob', type=float, default=0.0)
parser.add_argument('--iwm_noise_range', type=float, nargs=2, default=(0.05, 0.2))
parser.add_argument('--iwm_aug_ori', action='store_true', default=False)
parser.add_argument('--iwm_version', type=str, default='v1')
parser.add_argument('--iwm_aug_norm', action='store_true', default=False)
## landmarks
parser.add_argument('--num_landmarks', type=int, default=120)
parser.add_argument('--lm_window', type=int, default=3)
parser.add_argument('--lm_num_samples', type=int, default=10)

## tune type
parser.add_argument('--tune_type', default='fc', type=str)
parser.add_argument('--backbone_lr_scale', type=float, default=1.0)
parser.add_argument('--backbone_freeze', action='store_true', default=False)

parser.add_argument('--use_test', action='store_true', default=False)

parser.add_argument('--cache_rate', type=float, default=0.6)
parser.add_argument('--num_repeat_3d', type=int, default=2)

## sliding window inference
parser.add_argument('--overlap', type=float, default=0.5)
parser.add_argument('--infer_mode', type=str, default='constant')

parser.add_argument('--include_lateral', action='store_true', default=False)

## overlap
parser.add_argument('--overlap_grid_size', type=int, default=7)
parser.add_argument('--min_overlap', type=float, default=-1)
parser.add_argument('--policy_dim', type=int, default=5)

parser.add_argument('--stop_grad_conv1', action='store_true', default=False)
parser.add_argument('--stop_grad_norm1', action='store_true', default=False)
parser.add_argument('--mae_init_weights', action='store_true', default=False)
parser.add_argument('--flip_prob', type=float, default=0.5)

## var reg
parser.add_argument('--reg_weight', type=float, default=0.0)

## tencrop
parser.add_argument('--ten_crop', action='store_true', default=False)

parser.add_argument('--reverse_pred', action='store_true', default=False)

parser.add_argument('--unify_embed', action='store_true', default=False)


parser.add_argument('--cond_type', type=str, default='feat')

parser.add_argument('--num_aug_repeats', type=int, default=0)

parser.add_argument('--lt_label_group', type=str, default='all')

parser.add_argument('--seg_thres', type=float, default=0.5)


parser.add_argument('--unet_base_channels', type=int, default=32)

parser.add_argument('--seg_zero_thres', type=int, default=None)

parser.add_argument('--seg_out_indices', type=int, nargs='*', default=(11, 11, 11, 11, 11))