import torch
import torch.nn.functional as F
from .jepa import JEPA
from .roi import landmark_pooling
from .utils import apply_masks, repeat_interleave_batch, AllReduce
from .pos_embed import get_patches_feature



def reg_fn(z):
    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

class IWM(JEPA):
    def forward_context(self, imgs, aug_params, masks_enc, masks_pred):
        if self.list_mask:
            z_list = []
            for masks_enc_sub, masks_pred_sub in zip(masks_enc, masks_pred):
                z = self.encoder(imgs, masks_enc_sub)
                z = self.predictor(z, aug_params, masks_enc_sub, masks_pred_sub, is_mmb=True)
                z_list.append(z)
            return z_list
        else:
            z = self.encoder(imgs, masks_enc)
            z = self.predictor(z, aug_params, masks_enc, masks_pred, is_mmb=self.is_mmb)
            return z
    
    def forward(self, data, args):
        ((imgs, ori_imgs, aug_params), masks_enc, masks_pred) = data
        if args.iwm_disable:
            aug_params = torch.zeros_like(aug_params)
        h = self.forward_target(ori_imgs, masks_enc, masks_pred, args.target_last_k, args.target_norm_type)
        z = self.forward_context(imgs, aug_params, masks_enc, masks_pred)
        loss = self.loss_fn(z, h, args.loss_type)
        return loss



class IWM_Dual(JEPA):
    def forward_context_with_z(self, z_enc, aug_params, masks_enc, masks_pred, rel_pos=None):
        ''' Forward Context and return z_enc '''
        z = self.predictor(z_enc, aug_params, masks_enc, masks_pred, is_mmb=self.is_mmb, rel_pos_21=rel_pos)
        return z

    def forward(self, data, args):
        (image1, image_aug1, aug_params1, image2, image_aug2, aug_params2, rel_pos_21, rel_pos_12), masks_enc, masks_pred = data
        if args.iwm_disable:
            aug_params1 = torch.zeros_like(aug_params1)
            aug_params2 = torch.zeros_like(aug_params2)
        if args.rel_pos_disable:
            rel_pos_21 = None
            rel_pos_12 = None
        if args.reverse_pred:
            # pred augmented
            image1, image_aug1 = image_aug1, image1
            image2, image_aug2 = image_aug2, image2

        shuffle = torch.randperm(masks_enc[0].shape[0], device=image1.device) # shuffle masks
        masks_enc1, masks_pred1 = masks_enc, masks_pred
        masks_enc2 = [m[shuffle] for m in masks_enc]
        masks_pred2 = [m[shuffle] for m in masks_pred]
        
        ## Intra view mask modeling
        h1 = self.forward_target(image1, masks_enc1, masks_pred1, args.target_last_k, args.target_norm_type)
        h2 = self.forward_target(image2, masks_enc2, masks_pred2, args.target_last_k, args.target_norm_type)
        
        z1_enc = self.encoder(image_aug1, masks_enc1)
        z2_enc = self.encoder(image_aug2, masks_enc2)
        
        z11 = self.forward_context_with_z(z1_enc, aug_params1, masks_enc1, masks_pred1)
        z22 = self.forward_context_with_z(z2_enc, aug_params2, masks_enc2, masks_pred2)
        z21 = self.forward_context_with_z(z1_enc, aug_params1, masks_enc1, masks_pred2, rel_pos_21)
        z12 = self.forward_context_with_z(z2_enc, aug_params2, masks_enc2, masks_pred1, rel_pos_12)
        
        l11 = self.loss_fn(z11, h1, args.loss_type)
        l22 = self.loss_fn(z22, h2, args.loss_type)
        l21 = self.loss_fn(z21, h2, args.loss_type)
        l12 = self.loss_fn(z12, h1, args.loss_type)
        
        loss_intra = 0.5 * (l11 + l22)
        loss_extra = 0.5 * (l21 + l12)
        loss = loss_intra + args.extra_loss_weight * loss_extra
        if args.extra_mean:
            loss = loss / (1+args.extra_loss_weight)
        
        if args.extra_loss_weight > 0:
            pstd_z = reg_fn([z11, z22, z21, z12])  # predictor variance across patches
        else:
            pstd_z = reg_fn([z11, z22])
        loss_reg = torch.mean(F.relu(1.-pstd_z))
        loss = loss + args.reg_weight * loss_reg
        return dict(loss=loss, loss_intra=loss_intra.item(), loss_extra=loss_extra.item(),  pred_var=torch.mean(pstd_z).item())




from torch import nn
from copy import deepcopy
import torch.distributed

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
