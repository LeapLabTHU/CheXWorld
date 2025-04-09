# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import torch
import math
from monai.data import list_data_collate
_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        input_size=(96, 96, 96),
        patch_size=16,
        enc_mask_scale=(1.0, 1.0),
        pred_mask_scale=(0.15, 0.15),
        aspect_ratio=(0.3, 3.0),
        nenc=1, # this time, the predictor mask is always 1:npred, but repeat multiple times
        npred=8, 
        min_keep=10,
        max_keep=None,
        allow_overlap=False,
        rand_keep=False,
        merge=False,
        num_repeat=1,
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 3
        self.patch_size = patch_size
        self.height = self.width = self.depth = input_size[0] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.max_keep = max_keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes
        self.rand_keep = rand_keep
        self.merge = merge
        self.num_repeat = num_repeat
        
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        
        # -- Sample block scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * self.depth * mask_scale)
        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item() # change: sample two rand number
        
        min_ar, max_ar = aspect_ratio_scale
        min_ar, max_ar = math.log(min_ar), math.log(max_ar) # change:  use log ratio sampling following dinov2
        aspect_ratio = math.exp(min_ar + _rand * (max_ar - min_ar))        
        
        # -- Compute block height and width (given scale and aspect-ratio)
        d =  int(round(math.pow(max_keep, 1/3)))
        h = int(round(d * aspect_ratio))
        w = int(round(d / aspect_ratio))
        
        h = min(self.height, h)
        w = min(self.width, w)
        d = min(self.depth, d)
        size = torch.tensor([h, w, d])
        size = size[torch.randperm(3, generator=generator)].numpy() # shuffle to achieve symmetry
        return size

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w, d = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            i = torch.randint(0, self.height - h + 1, (1,))
            j = torch.randint(0, self.width - w + 1, (1,))
            k = torch.randint(0, self.depth - d + 1, (1,))
            mask = torch.zeros((self.height, self.width, self.depth), dtype=torch.int32)
            mask[i:i+h, j:j+w, k:k+d] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries) # remove parts not allowed
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep # mask is valid if it's large enough
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width, self.depth), dtype=torch.int32)
        mask_complement[i:i+h, j:j+w, k:k+d] = 0
        # --
        return mask, mask_complement # mask is indices, mask_complement is binary

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch) * self.num_repeat

        collated_batch = list_data_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width * self.depth # the smallest pred mask size
        min_keep_enc = self.height * self.width * self.depth # the smallest enc mask size
        for _ in range(B): # different mask for each element
            instance_masks_pred, instance_masks_enc = [], []
            for _ in range(self.nenc):
                masks_p, masks_C = [], []
                for _ in range(self.npred): # generate predict mask first, no unallowed regions
                    mask, mask_C = self._sample_block_mask(p_size)
                    masks_p.append(mask)
                    masks_C.append(mask_C)
                    
                if self.merge:
                    masks_p = [torch.unique(torch.cat(masks_p))]
                for mask in masks_p:
                    min_keep_pred = min(min_keep_pred, len(mask))
                instance_masks_pred.extend(masks_p)

                acceptable_regions = masks_C
                try:
                    if self.allow_overlap:
                        acceptable_regions= None
                except Exception as e:
                    logger.warning(f'Encountered exception in mask-generator {e}')

                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                min_keep_enc = min(min_keep_enc, len(mask))
                instance_masks_enc.append(mask)
            collated_masks_enc.append(instance_masks_enc)
            collated_masks_pred.append(instance_masks_pred)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)
        # mask_pred has nenc * npred
        if self.rand_keep:
            collated_masks_pred = [[rand_select(cm, min_keep_pred) for cm in cm_list] for cm_list in collated_masks_pred]
        else:
            collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred] # keep the smallest mask size
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        if self.rand_keep:
            collated_masks_enc = [[rand_select(cm, min_keep_enc) for cm in cm_list] for cm_list in collated_masks_enc]
        else:
            collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred


def rand_select(x, n):
    return x[torch.randperm(len(x))[:n]]