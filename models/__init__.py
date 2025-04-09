import torch
from torch import nn
from timm.layers import trunc_normal_
import logging
import math
from .jepa_vit import vit_predictor
from .jepa_vit_add import vit_predictor_conditioned
from .jepa import JEPA
from .jepa_add import *
from .finetune import FineTuner
from .models_vit import load_pretrained as load_pretrained_vit
from .unet_adapter_conv import UNetAdapter_4LayersConv, UNetAdapterConv

def mae_init_weights(model):
    # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    if hasattr(model, 'patch_embed'):
        w = model.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    if hasattr(model, 'cls_token'):
        torch.nn.init.normal_(model.cls_token, std=.02)
    if hasattr(model, 'mask_token'):
        torch.nn.init.normal_(model.mask_token, std=.02)

    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # initialize nn.Linear and LayerNorm
    model.apply(_init_weights)



def init_jepa_model(args, device, ssl_type):
    encoder = jepa_vit.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,
        )
    if args.drop_path > 0:
        logging.info('Warning: JEPA with encoder drop path!')
    if args.stop_grad_conv1:
        logging.info('Stop Grad Conv1!!!')
        encoder.patch_embed.proj.weight.requires_grad = False
        encoder.patch_embed.proj.bias.requires_grad = False
    if args.stop_grad_norm1:
        logging.info('Stop Grad Norm1!!!')
        encoder.blocks[0].norm1.weight.requires_grad = False
        encoder.blocks[0].norm1.bias.requires_grad = False
    args.extra = ('extra' in ssl_type) or ('dual' in ssl_type)
    if args.extra:
        logging.info('Model Extra Activated!!!!!')
    common_args = dict(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=args.pred_emb_dim,
        depth=args.pred_depth,
        num_heads=encoder.num_heads,
    )
    if 'iwm' in ssl_type:
        predictor = vit_predictor_conditioned(policy_dim=args.policy_dim if not args.iwm_disable else 0,policy_net_layers=3,**common_args, extra=args.extra, ssl_type=ssl_type, unify_embed=args.unify_embed, cond_type=args.cond_type)
    else:
        predictor = vit_predictor(extra=args.extra, ssl_type=ssl_type,**common_args)
    logging.info(f'Predictor: {predictor}')
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    if args.mae_init_weights:
        mae_init_weights(encoder)
        mae_init_weights(predictor)
        logging.info('Use MAE Init')
    else:
        for m in encoder.modules():
            init_weights(m)
        for m in predictor.modules():
            init_weights(m)
        logging.info('Use JEPA Init')

    logging.info(f'Mask Type: {args.mask_type}')
    # is_mbb = (args.mask_type != 'multiblock')
    # if is_mbb:
    #     logging.info('JEPA: Use MMB style mask!!!')
    list_mask = ('list' in args.mask_type)
    if list_mask:
        logging.info('Use List Style Mask!!')
    model_cls = {
        'iwm': IWM,
        'jepa': JEPA,
        'iwm_dual': IWM_Dual,
        'iwm_dual_easy': IWM_Dual,
    }[ssl_type]
    logging.info(f'SSL Model Type: {ssl_type}')
    if ssl_type == 'iwm' and args.iwm_disable:
        logging.info(f'IWM Disable Condition!!!')
    model = model_cls(
        encoder, predictor, 
        list_mask=list_mask
        )
    if args.pretrained:
        logging.info(f'Use dBOT-style teacher from {args.pretrained}')
        sd = torch.load(args.pretrained, map_location='cpu')['model']
        encoder_sd = {k.replace('target_encoder.', ''): v for k, v in sd.items() if k.startswith('target_encoder.')}
        model.target_encoder.load_state_dict(encoder_sd)
    model.to(device)
    return model


def build_transfer_model(args, seg=False):
    if args.with_cls_token:
        encoder = models_vit.__dict__[args.model](
            num_classes=0,
            drop_path_rate=args.drop_path,
            drop_rate=args.drop,
            global_pool=args.global_pool,
            img_size=args.input_size,
            # dynamic_img_size=True
        )
        if args.pretrained:
            load_pretrained_vit(args.pretrained, encoder)
    else:
        encoder = jepa_vit.__dict__[args.model](
            img_size=args.input_size,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            drop_path_uniform=args.drop_path_uniform
            )
        if args.pretrained:
            ckpt = torch.load(args.pretrained, map_location='cpu')
            sd = ckpt['model']
            if args.use_target:
                logging.info('Use Teacher')
                encoder_sd = {k.replace('target_encoder.', ''): v for k, v in sd.items() if k.startswith('target_encoder.')}
            else:
                logging.info('Use Student')
                encoder_sd = {k.replace('encoder.', ''): v for k, v in sd.items() if k.startswith('encoder.')}
            if args.input_size // args.patch_size != 14:
                pos_embed = encoder_sd['pos_embed']
                N, D = pos_embed.shape[1], pos_embed.shape[-1]
                pos_embed = nn.functional.interpolate(
                    pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), D).permute(0, 3, 1, 2),
                    scale_factor=math.sqrt((args.input_size // args.patch_size) ** 2 / N),
                    mode='bicubic', align_corners=False
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, D)
                encoder_sd['pos_embed'] = pos_embed
            if args.patch_size != 16:
                # resize patch embed kernel
                patch_embed = encoder_sd['patch_embed.proj.weight']
                patch_embed = torch.nn.functional.interpolate(patch_embed.float(), size=(args.patch_size, args.patch_size), mode='bicubic', align_corners=False)
                encoder_sd['patch_embed.proj.weight'] = patch_embed
            msg = encoder.load_state_dict(encoder_sd, strict=False)
            print('Missing:', msg.missing_keys)
    if seg:
        if args.patch_size == 8:
            model = UNetAdapter_4LayersConv(feature_model=encoder, num_classes=args.num_classes, base_channels=args.unet_base_channels)
        else:
            model = UNetAdapterConv(feature_model=encoder, num_classes=args.num_classes, base_channels=args.unet_base_channels)
        logging.info(f'Out indices: {args.seg_out_indices}')
        logging.info(f'Seg Model: {type(model)}')
    else:
        feature_dim = jepa_vit.VIT_EMBED_DIMS.get(args.model, 768)
        model = FineTuner(encoder, feature_dim=feature_dim, num_classes=args.num_classes, tune_type=args.tune_type, with_cls_token=args.with_cls_token)
    return model

