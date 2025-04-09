from functools import partial
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn
import os

def load_pretrained(pretrained, model):
    if pretrained.endswith('npz'):
        timm.models.vision_transformer._load_weights(model, pretrained) # npz load
    else:
        ckpt = torch.load(pretrained, map_location='cpu')
        if 'model' in ckpt:
            sd = ckpt['model']
        elif 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        # for clip models
        sd = {k.replace("vision_encoder.", ""): v for k, v in sd.items()}
        # for open_clip models
        sd = {k.replace("module.visual.trunk.", ""): v for k, v in sd.items()}
        # for moco
        sd = {k.replace("module.momentum_encoder.", ""): v for k, v in sd.items()}
        sd = checkpoint_filter_fn(sd, model)
        if 'head.weight' in sd:
            del sd['head.weight']
            del sd['head.bias']
        msg = model.load_state_dict(sd, strict=False)
        logging.info(f'Missing: {str(msg.missing_keys)}')
    logging.info(f'Load Pretrained {os.path.basename(pretrained)}')


def resize_pos_embed(pos_embed, grid_size):
    previous_dtype = pos_embed.dtype
    N = pos_embed.shape[1] - 1
    dim = pos_embed.shape[-1]
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    w0 = grid_size
    w0 = w0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), w0 / math.sqrt(N)),
        mode="bicubic",
    )
    assert int(w0) == patch_pos_embed.shape[-2]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    new_pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
    return new_pos_embed


def load_dinov2_model(args):
    import dinov2.models.vision_transformer as vits
    vit_kwargs = dict(
            img_size=224,
            patch_size=14,
            init_values=1.0e-05,
            ffn_layer="mlp" if args.model == 'vit_base' else "swiglufused",
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
    model = vits.__dict__[args.model](**vit_kwargs)

    sd = torch.load(args.finetune, map_location="cpu")
    sd['pos_embed'] = resize_pos_embed(sd['pos_embed'], 224 // 14)
    msg = model.load_state_dict(sd, strict=False)
    logging.info(f'Missing: {str(msg.missing_keys)}')
    logging.info(f'Load Pretrained {os.path.basename(args.finetune)}')
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(patch_size=16,**kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=14, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=14, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


