import torch.utils.data as data
import torch
import logging
from timm.data.distributed_sampler import RepeatAugSampler
from .xray_transform import build_transform_xray, build_seg_transform
from .chest_dset import MedFMC_Chest, ShenZhenCXR
from .vindr import VinDrNew
from .rsna import RSNA, RSNA_3Class, RSNASegmentDataset
from .mimic import MIMIC_Pretrain
from .world_model_transform import build_world_model_transform
from .xray_transform_add import build_world_model_dual_transform
from .nih import NIH
from .chexpert import CheXpert
from .siim import SIIM

def build_dataset(args, split='train', ten_crop=False):
    transform = build_transform_xray(args, is_train=(split == 'train'), ten_crop=ten_crop)
    if args.dataset == 'vindr_new':
        logging.info(f'Dataset VinDr New: Pct={args.data_pct}')
        dataset = VinDrNew(transform=transform, data_pct=args.data_pct, split=split, dataset_seed=args.dataset_seed)
    elif args.dataset == 'rsna':
        logging.info(f'Dataset RSNA: Pct={args.data_pct}')
        dataset = RSNA(transform=transform, data_pct=args.data_pct, split=split)
    elif args.dataset == 'rsna_3cls':
        dataset = RSNA_3Class(transform=transform, data_pct=args.data_pct, split=split)
    elif args.dataset == 'medfmc':
        logging.info(f'Dataset MedFMC: {args.fmc_shots} Shots, No.{args.fmc_exp_num}')
        dataset  = MedFMC_Chest(exp_num=args.fmc_exp_num, shots=args.fmc_shots, split=split, transform=transform)
    elif args.dataset == 'nih':
        logging.info(f'Dataset NIH: Pct={args.data_pct}')
        dataset = NIH(split=split, data_pct=args.data_pct, transform=transform, return_label=True)
    elif args.dataset == 'shenzhen':
        logging.info(f'Dataset ShenZhen')
        dataset = ShenZhenCXR(split=split, transform=transform)
    elif args.dataset == 'chex':
        dataset = CheXpert(transform=transform, return_label=True, data_pct=args.data_pct, split=split)
    else:
        raise NotImplementedError
    if args.dataset_cat > 1 and split == 'train':
        logging.info(f'Concat Dataset x{args.dataset_cat}')
        dataset = data.ConcatDataset([dataset for _ in range(args.dataset_cat)])
        dataset.num_classes = dataset.datasets[0].num_classes
    return dataset


def build_seg_dataset(args, split='train'):
    transform = build_seg_transform(args, is_train=(split=='train'))
    if args.dataset == 'siim':
        dataset = SIIM(transform=transform, split=split, data_pct=args.data_pct, balance=args.dataset_balance)
    elif args.dataset == 'rsna':
        dataset = RSNASegmentDataset(split=split, transform=transform, data_pct=args.data_pct)
    else:
        raise NotImplementedError
    if args.dataset_cat > 1 and split == 'train':
        logging.info(f'Concat Dataset x{args.dataset_cat}')
        dataset = data.ConcatDataset([dataset for _ in range(args.dataset_cat)])
        dataset.num_classes = dataset.datasets[0].num_classes
    return dataset

def build_pretrain_dataset(args, ssl_type):
    if ssl_type == 'iwm':
        logging.info('Build World Model Transform')
        transform = build_world_model_transform(args)
    elif 'iwm_dual' in ssl_type:
        transform = build_world_model_dual_transform(args)
    else:
        transform = build_transform_xray(args, is_train=True)
    
    if args.dataset == 'mimic':
        logging.info('Use MIMIC dataset')
        dataset = MIMIC_Pretrain(transform=transform, split='all', data_pct=args.data_pct, preload=args.preload_dataset, data_postfix=args.data_postfix, include_lateral=args.include_lateral)
    else:
        logging.info(f'Use Hybrid dataset {args.dataset}')
        d_list = []
        if 'nih' in args.dataset:
            d_list.append(NIH(transform=transform, data_pct=args.data_pct, return_label=False))
        if 'chex' in args.dataset:
            d_list.append(CheXpert(transform=transform, data_pct=args.data_pct, return_label=False, include_lateral=args.include_lateral))
        if 'mimic' in args.dataset:
            d_list.append(MIMIC_Pretrain(transform=transform, split='all', data_pct=args.data_pct, data_postfix=args.data_postfix, include_lateral=args.include_lateral))
        dataset = data.ConcatDataset(d_list)
        logging.info(f'Total Images: {len(dataset)}')
    return dataset

def get_loader(args, dataset, is_train=True, persistent_workers=True, custom_bs=None):
    if is_train:
        if args.num_aug_repeats:
            sampler = RepeatAugSampler(dataset, num_repeats=args.num_aug_repeats)
            logging.info(f'Use Repeated Aug Sampler: {args.num_aug_repeats}')
        else:
            num_tasks = args.world_size
            global_rank = args.rank
            sampler = data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
    else:
        sampler = data.SequentialSampler(dataset)
    
    this_bs = args.batch_size if is_train else args.batch_size * 2
    this_bs = custom_bs if custom_bs is not None else this_bs
    return data.DataLoader(
        dataset, sampler=sampler,
        batch_size=this_bs,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=is_train,
        persistent_workers=persistent_workers
    )

def get_infinite_loader(args, dataset, batch_size, custom_iter=None):
    # num_tasks = args.world_size
    # global_rank = args.rank
    
    num_iters = custom_iter if custom_iter is not None else args.num_iters
    num_samples = batch_size * num_iters
    if args.data_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.data_seed)
        logging.info(f'Use Data Seed {args.data_seed}')
    else:
        generator = None
        logging.info(f'No Data Seed')

    sampler = data.RandomSampler(dataset, replacement=(not args.data_no_replace), num_samples=num_samples, generator=generator)
    # sampler = DistributedProxySampler(sampler, num_replicas=num_tasks, rank=global_rank)
    sampler = data.BatchSampler(sampler, batch_size, drop_last=True)
    return data.DataLoader(
        dataset, 
        batch_sampler=sampler,         
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=True
    )