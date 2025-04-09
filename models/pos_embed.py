# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed



## From SiameseIM
def get_2d_sincos_pos_embed_relative(rel_pos, embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert rel_pos.shape[-1] == 6
    delta_i, delta_j, delta_h, delta_w = rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2], rel_pos[:, 3]
    relative_flip, flip_delta_j = rel_pos[:, 4], rel_pos[:, 5]
    
    delta_i = delta_i * grid_size
    delta_j = delta_j * grid_size
    flip_delta_j = flip_delta_j * grid_size
    grid_h = torch.arange(grid_size, dtype=rel_pos.dtype, device=rel_pos.device)
    grid_w = torch.arange(grid_size, dtype=rel_pos.dtype, device=rel_pos.device)
    raw_grid_h, raw_grid_w = torch.meshgrid(grid_h, grid_w)

    raw_grid_h = raw_grid_h + 0.5
    raw_grid_w = raw_grid_w + 0.5
    grid_h = torch.einsum('b,n->bn', delta_h, raw_grid_h.flatten()) + delta_i.unsqueeze(-1)
    grid_w = torch.einsum('b,n->bn', delta_w, raw_grid_w.flatten()) + delta_j.unsqueeze(-1)

    flip_grid_w = -torch.einsum('b,n->bn', [delta_w, raw_grid_w.flatten()]) + flip_delta_j[:, None]
    relative_flip = relative_flip.float().unsqueeze(-1)
    grid_w = relative_flip * flip_grid_w + (1-relative_flip) * grid_w
    grid_w = grid_w - 0.5
    grid_h = grid_h - 0.5

    omega = torch.arange(embed_dim//4, dtype=rel_pos.dtype, device=rel_pos.device) / (embed_dim/4)
    omega = 1. / (10000**omega)
    out_h = torch.einsum('bn,c->bnc', [grid_h, omega])
    out_w = torch.einsum('bn,c->bnc', [grid_w, omega])
    out_scale_h = torch.einsum('b,c->bc', [10*torch.log(delta_h), omega]).unsqueeze(1).expand(-1, out_h.shape[1], -1)
    out_scale_w = torch.einsum('b,c->bc', [10*torch.log(delta_w), omega]).unsqueeze(1).expand(-1, out_h.shape[1], -1)
    pos_embed = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w),
                           torch.sin(out_scale_h), torch.cos(out_scale_h),
                           torch.sin(out_scale_w), torch.cos(out_scale_w),], dim=2).detach()

    return pos_embed




def get_2d_sincos_pos_embed_relative_easy(rel_pos, embed_dim, grid_size, cls_token=False):
    """
    Same scale prediction
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert rel_pos.shape[-1] == 4
    delta_i, delta_j = rel_pos[:, 0], rel_pos[:, 1]
    relative_flip, flip_delta_j = rel_pos[:, 2], rel_pos[:, 3]
    delta_h, delta_w = torch.ones_like(delta_i), torch.ones_like(delta_i)
    
    delta_i = delta_i * grid_size
    delta_j = delta_j * grid_size
    flip_delta_j = flip_delta_j * grid_size
    grid_h = torch.arange(grid_size, dtype=rel_pos.dtype, device=rel_pos.device)
    grid_w = torch.arange(grid_size, dtype=rel_pos.dtype, device=rel_pos.device)
    raw_grid_h, raw_grid_w = torch.meshgrid(grid_h, grid_w)

    raw_grid_h = raw_grid_h + 0.5
    raw_grid_w = raw_grid_w + 0.5
    grid_h = torch.einsum('b,n->bn', delta_h, raw_grid_h.flatten()) + delta_i.unsqueeze(-1)
    grid_w = torch.einsum('b,n->bn', delta_w, raw_grid_w.flatten()) + delta_j.unsqueeze(-1)

    flip_grid_w = -torch.einsum('b,n->bn', [delta_w, raw_grid_w.flatten()]) + flip_delta_j[:, None]
    relative_flip = relative_flip.float().unsqueeze(-1)
    grid_w = relative_flip * flip_grid_w + (1-relative_flip) * grid_w
    grid_w = grid_w - 0.5
    grid_h = grid_h - 0.5

    omega = torch.arange(embed_dim//4, dtype=rel_pos.dtype, device=rel_pos.device) / (embed_dim/4)
    omega = 1. / (10000**omega)
    out_h = torch.einsum('bn,c->bnc', [grid_h, omega])
    out_w = torch.einsum('bn,c->bnc', [grid_w, omega])
    # out_scale_h = torch.einsum('b,c->bc', [10*torch.log(delta_h), omega]).unsqueeze(1).expand(-1, out_h.shape[1], -1)
    # out_scale_w = torch.einsum('b,c->bc', [10*torch.log(delta_w), omega]).unsqueeze(1).expand(-1, out_h.shape[1], -1)
    pos_embed = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)], dim=2).detach()
    
    return pos_embed




def get_3d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_size, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)  # order of meshgrid is very important for indexing as [d,h,w]

    h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim/6)*2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed



def get_3d_sincos_pos_embed_relative_easy(rel_pos, embed_dim, grid_size, cls_token=False):
    """
    Same scale prediction
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert rel_pos.shape[-1] == 3
    delta_i, delta_j, delta_k = rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2]

    delta_d, delta_h, delta_w = torch.ones_like(delta_i), torch.ones_like(delta_i),  torch.ones_like(delta_i)
    
    delta_i = delta_i * grid_size
    delta_j = delta_j * grid_size
    delta_k = delta_k * grid_size
    
    grid_d = torch.arange(grid_size, dtype=rel_pos.dtype, device=rel_pos.device)
    grid_h = torch.arange(grid_size, dtype=rel_pos.dtype, device=rel_pos.device)
    grid_w = torch.arange(grid_size, dtype=rel_pos.dtype, device=rel_pos.device)
    raw_grid_d, raw_grid_h, raw_grid_w = torch.meshgrid(grid_d, grid_h, grid_w)

    raw_grid_d = raw_grid_d + 0.5
    raw_grid_h = raw_grid_h + 0.5
    raw_grid_w = raw_grid_w + 0.5
    grid_d = torch.einsum('b,n->bn', delta_d, raw_grid_d.flatten()) + delta_i.unsqueeze(-1)
    grid_h = torch.einsum('b,n->bn', delta_h, raw_grid_h.flatten()) + delta_j.unsqueeze(-1)
    grid_w = torch.einsum('b,n->bn', delta_w, raw_grid_w.flatten()) + delta_k.unsqueeze(-1)

    grid_d = grid_d - 0.5
    grid_w = grid_w - 0.5
    grid_h = grid_h - 0.5

    omega = torch.arange(embed_dim//6, dtype=rel_pos.dtype, device=rel_pos.device) / (embed_dim/6)
    omega = 1. / (10000**omega)
    out_d = torch.einsum('bn,c->bnc', [grid_d, omega])
    out_h = torch.einsum('bn,c->bnc', [grid_h, omega])
    out_w = torch.einsum('bn,c->bnc', [grid_w, omega])

    pos_embed = torch.cat([torch.sin(out_d), torch.cos(out_d), torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)], dim=2).detach()
    
    return pos_embed

from einops import rearrange
import torch.nn.functional as F

def get_patches_feature(h, pos, out_grid_size):
    B, L, D = h.shape
    grid_size = int(L ** 0.5)
    h = rearrange(h, 'b (h w) c -> b c h w', h=grid_size, w=grid_size)
    
    theta = torch.zeros((B, 2, 3), device=h.device)
    x1, x2, y1, y2, = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
    delta_h, delta_w = pos[:, 4], pos[:, 5]
    
    theta[:, 0, 0], theta[:, 1, 1] = delta_h, delta_w
    theta[:, 0, 2], theta[:, 1, 2] = -1 + (x1 + x2), -1 + (y1 + y2)

    grid = F.affine_grid(theta.float(), torch.Size((B, D, out_grid_size, out_grid_size)))
    patches = F.grid_sample(h, grid)  # B, D, G, G
    patches = rearrange(patches, 'b c h w -> b (h w) c')
    return patches



def fourier_encode(x, max_freq=128, num_bands=32):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * np.pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x