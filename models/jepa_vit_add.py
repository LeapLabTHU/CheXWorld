import math
from functools import partial
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_embed import *
from .utils import apply_masks, repeat_interleave_batch, trunc_normal_
from .jepa_vit import VisionTransformerPredictor



class VisionTransformerPredictorConditioned(VisionTransformerPredictor):
    def __init__(self, policy_dim=4, policy_net_layers=3, unify_embed=False, cond_type='feat', **kwargs):
        super().__init__(**kwargs)
        self.policy_dim = policy_dim
        self.cond_type = cond_type
        if policy_dim > 0:
            logging.info(f'Condition Type: {cond_type}')
            if cond_type == 'feat' or cond_type == 'feat_res':
                layers = [nn.Linear(self.predictor_embed_dim+self.policy_dim, self.predictor_embed_dim)]
                for _ in range(policy_net_layers-1):
                    layers.extend([nn.ReLU(), nn.Linear(self.predictor_embed_dim, self.predictor_embed_dim)])
                self.policy_net = nn.Sequential(*layers)
            elif cond_type == 'feat_silu':
                layers = [nn.Linear(self.predictor_embed_dim+self.policy_dim, self.predictor_embed_dim)]
                for _ in range(policy_net_layers-1):
                    layers.extend([nn.SiLU(), nn.Linear(self.predictor_embed_dim, self.predictor_embed_dim)])
                self.policy_net = nn.Sequential(*layers)
            elif cond_type == 'token':
                dummy_aug_param = torch.rand(1, policy_dim)
                encoding = fourier_encode(dummy_aug_param).view(-1)
                logging.info(f'Freq embed dim: {encoding.shape[-1]}')
                self.policy_net = nn.Sequential(
                    nn.Linear(encoding.shape[-1], 4 * self.predictor_embed_dim),
                    nn.SiLU(),
                    nn.Linear(4 * self.predictor_embed_dim, self.predictor_embed_dim),
                )
            elif cond_type == 'token_bare':
                self.policy_net = nn.Sequential(
                    nn.Linear(policy_dim, self.predictor_embed_dim),
                    nn.SiLU(),
                    nn.Linear(self.predictor_embed_dim, self.predictor_embed_dim),
                )
            else:
                raise NotImplementedError
        else:
            logging.info('No policy net!!!!')
            self.policy_net = None
        if unify_embed:
            logging.info('Predictor Unify Embed!')
            rel_pos_21 = torch.zeros((1, 4), dtype=torch.float)
            pos_embs = get_2d_sincos_pos_embed_relative_easy(rel_pos_21, self.predictor_pos_embed.shape[-1],
                                                        int(self.num_patches ** .5)) #[B, L, H]
            self.predictor_pos_embed.data.copy_(pos_embs)
    
    def forward(self, x, aug_params, masks_x, masks, is_mmb=False,rel_pos_21=None):
        assert (masks is not None), 'Cannot run predictor without mask indices'
        # assert aug_params.shape[-1] == self.policy_dim, f'Policy dim mismatch! {aug_params.shape[-1]} vs {self.policy_dim}'
        no_input_mask = (masks_x is None)
        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens, original ones
        if no_input_mask:
            x += self.interpolate_pos_encoding(x, self.predictor_pos_embed)
        else:
            x_pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embs, masks_x) #[nenc*B]

        _, N_ctxt, D = x.shape

        if self.extra:
            # use unified embedding for mask token
            if self.is_3d:
                if rel_pos_21 is None:
                    rel_pos_21 = torch.zeros((B, 3), dtype=torch.float, device=x.device)
                pos_embs = get_3d_sincos_pos_embed_relative_easy(rel_pos_21, self.predictor_pos_embed.shape[-1], round(self.num_patches**(1/3)))
            elif 'easy' in self.ssl_type:
                if rel_pos_21 is None:
                    rel_pos_21 = torch.zeros((B, 4), dtype=torch.float, device=x.device)
                pos_embs = get_2d_sincos_pos_embed_relative_easy(rel_pos_21, self.predictor_pos_embed.shape[-1],
                                                        int(self.num_patches ** .5)) #[B, L, H]
            else:
                if rel_pos_21 is None:
                    rel_pos_21 = torch.zeros((B, 6), dtype=torch.float, device=x.device)
                    rel_pos_21[:, 2:4] = 1 # delta h,w
                pos_embs = get_2d_sincos_pos_embed_relative(rel_pos_21, self.predictor_pos_embed.shape[-1],
                                                        int(self.num_patches ** .5)) #[B, L, H]
                pos_embs = self.predictor_pos_mlp(pos_embs.float())
        else:
            raise NotImplementedError
            # pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        # -- concat mask tokens to x
        pos_embs = apply_masks(pos_embs, masks) #[npred*B]
        if not is_mmb:
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x)) #[npred*nenc*B]
        else:
            pass #[nenc*npred*B]
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        
        # concat aug params
        if self.policy_net is not None:
            if 'feat' in self.cond_type:
                aug_params = aug_params.unsqueeze(1).repeat(pred_tokens.shape[0]//B, pred_tokens.shape[1], 1)
                if self.cond_type == 'feat_res':
                    pred_tokens = pred_tokens + self.policy_net(torch.cat([pred_tokens, aug_params], dim=-1))
                else:
                    pred_tokens = torch.cat([pred_tokens, aug_params], dim=-1)
                    pred_tokens = self.policy_net(pred_tokens)
            elif self.cond_type == 'token':
                aug_embedding = self.policy_net(fourier_encode(aug_params).view(B, -1)).unsqueeze(1)
                pred_tokens = torch.cat([pred_tokens, aug_embedding], dim=1)
            elif self.cond_type == 'token_bare':
                aug_embedding = self.policy_net(aug_params).unsqueeze(1)
                pred_tokens = torch.cat([pred_tokens, aug_embedding], dim=1)
        
        if not is_mmb:
            x = x.repeat(len(masks), 1, 1) #[npred*nenc*B]
        else:
            x = repeat_interleave_batch(x, B, repeat=len(masks) // len(masks_x)) #[nenc*npred*B]
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        if 'token' in self.cond_type:
            x = x[:, :-1]
        x = self.predictor_proj(x)

        return x


class VisionTransformerPredictorCodebook(VisionTransformerPredictor):
    def __init__(self, num_lms=120, **kwargs):
        super().__init__(**kwargs)
        self.lm_embeddings = nn.Parameter(torch.zeros(num_lms, self.predictor_embed_dim), requires_grad=True)
        trunc_normal_(self.lm_embeddings, std=self.init_std)
    
    def get_embeddings(self, lm_indices):
        B = lm_indices.shape[0]
        lm_indices = lm_indices.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim)
        return torch.gather(self.lm_embeddings.expand(B, *self.lm_embeddings.shape), 1, lm_indices)

    def forward(self, x, masks_x, masks, is_mmb=False, rel_pos_21=None, lm_indices=None):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x) #[nenc*B]

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        
        if rel_pos_21 is not None:
            pos_embs = get_2d_sincos_pos_embed_relative(*rel_pos_21, self.predictor_pos_embed.shape[-1],
                                                    int(self.num_patches ** .5))
            pos_embs = self.predictor_pos_mlp(pos_embs.float())
        else:
            pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)

        pos_embs = apply_masks(pos_embs, masks) #[npred*B]
        if not is_mmb:
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x)) #[npred*nenc*B]
        else:
            pass #[nenc*npred*B]
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        
        # lm_pos_embs = self.lm_embeddings(lm_indices)
        lm_pos_embs = self.get_embeddings(lm_indices)
        lm_pred_tokens = self.mask_token.repeat(lm_pos_embs.size(0), lm_pos_embs.size(1), 1)
        lm_pred_tokens += lm_pos_embs
        pred_tokens = torch.cat([pred_tokens, lm_pred_tokens], dim=1)
        
        if not is_mmb:
            x = x.repeat(len(masks), 1, 1) #[npred*nenc*B]
        else:
            x = repeat_interleave_batch(x, B, repeat=len(masks) // len(masks_x)) #[nenc*npred*B]
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1]
        N = pos_embed.shape[1]
        if npatch == N:
            return pos_embed
        # class_emb = pos_embed[:, 0]
        # pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return  pos_embed




class VisionTransformerPredictorConditionedOverlap(VisionTransformerPredictor):
    def __init__(self, policy_dim=4, policy_net_layers=3, overlap_grid_size=7, **kwargs):
        super().__init__(**kwargs)
        self.policy_dim = policy_dim
        layers = [nn.Linear(self.predictor_embed_dim+self.policy_dim, self.predictor_embed_dim)]
        for _ in range(policy_net_layers-1):
            layers.extend([nn.ReLU(), nn.Linear(self.predictor_embed_dim, self.predictor_embed_dim)])
        self.policy_net = nn.Sequential(*layers)
        self.overlap_grid_size = overlap_grid_size
    
    def forward(self, x, aug_params, masks_x, masks, is_mmb=False,rel_pos_21=None):
        # assert (masks is not None), 'Cannot run predictor without mask indices'
        assert aug_params.shape[-1] == self.policy_dim, f'Policy dim mismatch! {aug_params.shape[-1]} vs {self.policy_dim}'
        no_input_mask = (masks_x is None)
        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens, original ones
        if no_input_mask:
            x += self.interpolate_pos_encoding(x, self.predictor_pos_embed)
        else:
            x_pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embs, masks_x) #[nenc*B]

        _, N_ctxt, D = x.shape

        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        if rel_pos_21 is not None:
            # no pred mask, but a interpolated grid
            pos_embs = get_patches_feature(pos_embs, rel_pos_21, out_grid_size=self.overlap_grid_size)
        else:
            # -- concat mask tokens to x
            pos_embs = apply_masks(pos_embs, masks) #[npred*B]
            if not is_mmb:
                pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x)) #[npred*nenc*B]
            else:
                pass #[nenc*npred*B]
        

        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        
        # concat aug params
        aug_params = aug_params.unsqueeze(1).repeat(pred_tokens.shape[0]//B, pred_tokens.shape[1], 1)
        pred_tokens = torch.cat([pred_tokens, aug_params], dim=-1)
        pred_tokens = self.policy_net(pred_tokens)
        
        if not is_mmb:
            x = x.repeat(len(masks), 1, 1) #[npred*nenc*B]
        else:
            x = repeat_interleave_batch(x, B, repeat=len(masks) // len(masks_x)) #[nenc*npred*B]
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x

def vit_predictor_conditioned(**kwargs):
    model = VisionTransformerPredictorConditioned(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

def vit_predictor_conditioned_overlap(**kwargs):
    model = VisionTransformerPredictorConditionedOverlap(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

def vit_predictor_coodbook(**kwargs):
    model = VisionTransformerPredictorCodebook(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model