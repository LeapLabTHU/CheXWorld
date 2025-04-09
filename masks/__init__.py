from .multiblock import MaskCollator as MBCollator
from .multi_multiblock import MaskCollator as MMBCollator
from .multi_multiblock_v2 import MaskCollator as MMBV2Collator
from .multi_multiblock_list import MaskCollator as MMBLCollator
from .extra_multiblock import MaskCollator  as EMBCollator
from .random import MaskCollator as RandomCollator
from .data2vec import MaskCollator as D2VCollator
from .multi_multiblock_3d import MaskCollator as MMB3DCollator

def build_mask_collator(args):
    # if args.mask_type == 'multiblock':
    #     collator = MBCollator(
    #         input_size=args.input_size,
    #         patch_size=args.patch_size,
    #         pred_mask_scale=args.pred_mask_scale,
    #         enc_mask_scale=args.enc_mask_scale,
    #         aspect_ratio=(0.75, 1.5),
    #         nenc=args.mask_nenc,
    #         npred=args.mask_npred,
    #         allow_overlap=False,
    #         min_keep=args.mask_min_keep,
    #         max_keep=args.mask_max_keep,
    #         rand_keep=args.mask_rand_keep
    #         )
    if args.mask_type == 'multi_multiblock':
        collator = MMBCollator(
            input_size=args.input_size,
            patch_size=args.patch_size,
            pred_mask_scale=args.pred_mask_scale,
            enc_mask_scale=args.enc_mask_scale,
            aspect_ratio=(0.75, 1.5),
            nenc=args.mask_nenc,
            npred=args.mask_npred,
            allow_overlap=False,
            min_keep=args.mask_min_keep,
            max_keep=args.mask_max_keep,
            rand_keep=args.mask_rand_keep,
            merge=args.mask_merge
            )
    elif args.mask_type == 'multi_multiblock_v2':
        collator = MMBV2Collator(
            input_size=args.input_size,
            patch_size=args.patch_size,
            pred_mask_scale=args.pred_mask_scale,
            enc_mask_scale=args.enc_mask_scale,
            aspect_ratio=(0.75, 1.5),
            nenc=args.mask_nenc,
            npred=args.mask_npred,
            allow_overlap=False,
            min_keep=args.mask_min_keep,
            max_keep=args.mask_max_keep,
            rand_keep=args.mask_rand_keep,
            merge=args.mask_merge
            )
    elif args.mask_type == 'multi_multiblock_list':
        collator = MMBLCollator(
            input_size=args.input_size,
            patch_size=args.patch_size,
            pred_mask_scale=args.pred_mask_scale,
            enc_mask_scale=args.enc_mask_scale,
            aspect_ratio=(0.75, 1.5),
            nenc=args.mask_nenc,
            npred=args.mask_npred,
            allow_overlap=False,
            min_keep=args.mask_min_keep,
            max_keep=args.mask_max_keep,
            rand_keep=args.mask_rand_keep,
            merge=args.mask_merge
            )
    elif args.mask_type == 'multi_multiblock_3d':
        collator = MMB3DCollator(
            input_size=args.input_size,
            patch_size=args.patch_size,
            pred_mask_scale=args.pred_mask_scale,
            enc_mask_scale=args.enc_mask_scale,
            aspect_ratio=(0.75, 1.5),
            nenc=args.mask_nenc,
            npred=args.mask_npred,
            allow_overlap=False,
            min_keep=args.mask_min_keep,
            max_keep=args.mask_max_keep,
            rand_keep=args.mask_rand_keep,
            merge=args.mask_merge,
            num_repeat=args.num_repeat_3d
        )
    elif args.mask_type == 'extra_multiblock':
        collator = EMBCollator(
            input_size=args.input_size,
            patch_size=args.patch_size,
            pred_mask_scale=args.pred_mask_scale,
            enc_mask_scale=args.enc_mask_scale,
            aspect_ratio=(0.75, 1.5),
            nenc=args.mask_nenc,
            npred=args.mask_npred,
            min_keep=args.mask_min_keep,
            max_keep=args.mask_max_keep,
            rand_keep=args.mask_rand_keep,
            )
    elif args.mask_type == 'data2vec':
        collator = D2VCollator(
            input_size=args.input_size,
            patch_size=args.patch_size,
            nenc=args.mask_nenc
        )
    elif args.mask_type == 'random':
        collator = RandomCollator(
            ratio=(0.75, 0.75),
            input_size=args.input_size,
            patch_size=args.patch_size
        )
    else:
        raise NotImplementedError
    return collator