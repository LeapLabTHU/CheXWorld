import torch
from typing import Tuple
import math

from multiprocessing import Value

from logging import getLogger
from .blockwise import MaskingGenerator
import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        nenc=1,
        mask_prob=0.8,
        mask_length=3,
        mask_prob_adjust=0.07,
        inverse_mask=True
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.nenc = nenc
        self.num_tokens = self.height * self.width
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_prob_adjust = mask_prob_adjust
        self.inverse_mask = inverse_mask
        if not self.inverse_mask:
            self.block_generator = MaskingGenerator((self.height, self.width))


    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)
        
        if self.inverse_mask:
            mask = compute_block_mask_2d(
                            shape=(B*self.nenc, self.num_tokens),
                            mask_prob=self.mask_prob,
                            mask_length=self.mask_length,
                            mask_prob_adjust=self.mask_prob_adjust,
                            inverse_mask=True,
                            require_same_masks=True,
                        )
        else:
            mask = torch.stack([torch.from_numpy(self.block_generator()).view(-1) for _ in range(B*self.nenc)], dim=0)
            mask = 1 - mask # 0 = masked
        
        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        len_keep = self.num_tokens - mask[0].sum()
        ids_keep = ids_shuffle[:, :len_keep]
        ids_pred = ids_shuffle[:, len_keep:]
        
        collated_masks_pred, collated_masks_enc = [], []
        for bidx in range(B):
            collated_masks_enc.append([ids_keep[bidx*self.nenc+n] for n in range(self.nenc)])
            collated_masks_pred.append([ids_pred[bidx*self.nenc+n] for n in range(self.nenc)])

        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred

def compute_block_mask_2d(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False,
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
) -> torch.Tensor:

    assert mask_length > 1

    B, L = shape

    d = int(L**0.5)

    if inverse_mask:
        mask_prob = 1 - mask_prob

    if non_overlapping:
        sz = math.ceil(d / mask_length)
        inp_len = sz * sz

        inp = torch.zeros((B, 1, sz, sz))
        w = torch.ones((1, 1, mask_length, mask_length))

        mask_inds = torch.multinomial(
            1 - inp.view(B, -1),
            int(inp_len * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
            replacement=False,
        )
        inp.view(B, -1).scatter_(1, mask_inds, 1)

        mask = torch.nn.functional.conv_transpose2d(inp, w, stride=mask_length).squeeze(
            1
        )
        if mask.size(-1) > d:
            mask = mask[..., :d, :d]
    else:
        mask = torch.zeros((B, d, d))
        mask_inds = torch.randint(
            0,
            L,
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / mask_length**2)
                    * (1 + mask_dropout)
                ),
            ),
        )
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [], [])

        offset = mask_length // 2
        for i in range(mask_length):
            for j in range(mask_length):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d - 1)

        mask[(i0, i1, i2)] = 1

    def get_nbs(b, m, w):
        all_nbs = torch.nn.functional.conv2d(m.unsqueeze(1), w, padding="same")
        all_nbs = all_nbs.clamp_max_(1).view(b, -1)
        return all_nbs

    if require_same_masks and expand_adjcent:
        w = torch.zeros((1, 1, 3, 3))
        w[..., 0, 1] = 1
        w[..., 2, 1] = 1
        w[..., 1, 0] = 1
        w[..., 1, 2] = 1

        all_nbs = get_nbs(B, mask, w)

    mask = mask.reshape(B, -1)

    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * (mask_prob))
        target_len = int(final_target_len * (1 + mask_dropout))

        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.view(1, d, d), w).flatten()

                cands = (1 - m + nbs) > 1
                cand_sz = int(cands.sum().item())

                assert cand_sz > 0, f"{nbs} {cand_sz}"

                to_mask = torch.multinomial(
                    cands.float(), min(cand_sz, int(target_len - n)), replacement=False
                )
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1

            if n > final_target_len:
                to_unmask = torch.multinomial(
                    m, int(n - final_target_len), replacement=False
                )
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(
                    (1 - m), int(final_target_len - n), replacement=False
                )
                m[to_mask] = 1

    if inverse_mask:
        mask = 1 - mask

    return mask


# if __name__ == '__main__':
#     T = 14*14
#     mask = compute_block_mask_2d(
#                         shape=(1, T),
#                         mask_prob=0.8,
#                         mask_length=3,
#                         mask_prob_adjust=0.07,
#                         inverse_mask=True,
#                         require_same_masks=True,
#                     )
#     mask = mask.to(torch.uint8)
#     print(mask.shape, mask.sum())
#     ids_shuffle = mask.argsort(dim=1)
#     ids_restore = ids_shuffle.argsort(dim=1)
#     len_keep = T - mask[0].sum()
#     ids_keep = ids_shuffle[:, :len_keep]
#     mask1 = torch.ones_like(mask)
#     mask1[0, ids_keep] = 0
#     print(mask1)
#     print(mask)
#     print(ids_keep)

        