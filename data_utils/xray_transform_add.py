import math
import random
import logging
from PIL import ImageFilter, ImageOps
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from .xray_transform import get_mean_std, build_aug_trans

class SingleRandomResizedCrop(transforms.RandomResizedCrop):
    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, width

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w, width = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), i, j, h, w, width
    

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img):
        if torch.rand(1) < self.p:
            return F.hflip(img), True
        return img, False
    

class ParameterizedCrop(object):
    def __init__(self, global_size=224, global_scale=(0.3, 1.0), local_size=96, local_scale=(0.05, 0.3), post_trans=None,):
        self.global_rrc = SingleRandomResizedCrop(global_size, scale=global_scale, interpolation=3)
        self.local_rrc = SingleRandomResizedCrop(local_size, scale=local_scale, interpolation=3)
        self.random_flip = RandomHorizontalFlip()
        self.format_transform = post_trans
        logging.info(f'Parameterized Crop with global {global_size}@{global_scale}, local {local_size}@({local_scale})')

    def __call__(self, image):
        spatial_image1, flip1 = self.random_flip(image)
        spatial_image2, flip2 = self.random_flip(image)
        spatial_image1, i1, j1, h1, w1, W = self.local_rrc(spatial_image1) # local to global prediction
        spatial_image2, i2, j2, h2, w2, W = self.global_rrc(spatial_image2)
        relative_flip = (flip1 and not flip2) or (flip2 and not flip1)
        rel_pos_21 = ((i2-i1)/h1, (j2-j1)/w1, h2/h1, w2/w1, relative_flip, (W-j1-j2)/w1)
        return self.format_transform(spatial_image1), self.format_transform(spatial_image2), rel_pos_21

    def __repr__(self):
        repr = "(ParameterizedCrop,\n"
        repr += "  transform = %s,\n" % str(self.global_rrc) + str(self.local_rrc) + str(self.random_flip) + str(self.format_transform)
        repr += ")"
        return repr




class ParameterizedCropMulti(object):
    def __init__(self, num_local=1, global_size=224, global_scale=(0.3, 1.0), local_size=96, local_scale=(0.05, 0.3), post_trans=None,):
        self.global_rrc = SingleRandomResizedCrop(global_size, scale=global_scale, interpolation=3)
        self.local_rrc = SingleRandomResizedCrop(local_size, scale=local_scale, interpolation=3)
        self.random_flip = RandomHorizontalFlip()
        self.format_transform = post_trans
        self.num_local = num_local
        logging.info(f'Parameterized Crop with global {global_size}@{global_scale}, local {local_size}@{local_scale}x{num_local}')

    def __call__(self, image):
        global_image, global_flip = self.random_flip(image)
        global_image, ig, jg, hg, wg, W = self.global_rrc(global_image)
        global_image = self.format_transform(global_image)
        local_images = []
        local_pos = []
        for _ in range(self.num_local):
            local_image, local_flip = self.random_flip(image)
            local_image, ic, jc, hc, wc, W = self.local_rrc(local_image) # local to global prediction
            relative_flip = int((global_flip and not local_flip) or (local_flip and not global_flip))
            rel_pos_21 = torch.tensor([(ig-ic)/hc, (jg-jc)/wc, hg/hc, wg/wc, relative_flip, (W-jg-jc)/wc], dtype=torch.float)
            local_images.append(self.format_transform(local_image))
            local_pos.append(rel_pos_21)
        local_images = torch.stack(local_images, dim=0) #[N, C, H, W]
        local_pos = torch.stack(local_pos, dim=0) #[N, 6]
        return global_image, local_images,  local_pos

    def __repr__(self):
        repr = "(ParameterizedCrop,\n"
        repr += "  transform = %s,\n" % str(self.global_rrc) + str(self.local_rrc) + str(self.random_flip) + str(self.format_transform)
        repr += ")"
        return repr


from .world_model_transform import build_parameterized_aug

class ParameterizedCropWithAug(object):
    def __init__(self, num_local=1, global_size=224, global_scale=(0.3, 1.0), local_size=96, local_scale=(0.05, 0.3), post_trans=None, aug_trans=None, aug_trans_param=None):
        self.global_rrc = SingleRandomResizedCrop(global_size, scale=global_scale, interpolation=3)
        self.local_rrc = SingleRandomResizedCrop(local_size, scale=local_scale, interpolation=3)
        self.aug_trans = aug_trans
        self.aug_trans_param = aug_trans_param
        self.random_flip = RandomHorizontalFlip()
        self.format_transform = post_trans
        self.num_local = num_local
        logging.info(f'Parameterized Crop with global {global_size}@{global_scale}, local {local_size}@{local_scale}x{num_local}')

    def __call__(self, image):
        image = self.aug_trans(image) # color transform on original image
        
        global_image, global_flip = self.random_flip(image)
        global_image, ig, jg, hg, wg, W = self.global_rrc(global_image)
        global_image_aug, global_aug_params = self.aug_trans_param(global_image, aug_only=True)
        
        global_image = self.format_transform(global_image)
        global_image_aug = self.format_transform(global_image_aug)
        
        local_images = []
        local_pos = []
        local_aug_params_list = []
        for _ in range(self.num_local):
            local_image, local_flip = self.random_flip(image)
            local_image, ic, jc, hc, wc, W = self.local_rrc(local_image) # local to global prediction
            local_image, local_aug_params = self.aug_trans_param(local_image, aug_only=True) # aug on local image
            
            relative_flip = int((global_flip and not local_flip) or (local_flip and not global_flip))
            rel_pos_21 = torch.tensor([(ig-ic)/hc, (jg-jc)/wc, hg/hc, wg/wc, relative_flip, (W-jg-jc)/wc], dtype=torch.float)
            
            local_images.append(self.format_transform(local_image))
            local_pos.append(rel_pos_21)
            local_aug_params_list.append(local_aug_params)

        local_images = torch.stack(local_images, dim=0) #[N, C, H, W]
        local_pos = torch.stack(local_pos, dim=0) #[N, 6]
        local_aug_params_list = torch.stack(local_aug_params_list, dim=0) #[N, A]
        return global_image, global_image_aug, global_aug_params, local_images, local_pos, local_aug_params_list

    def __repr__(self):
        repr = "(ParameterizedCrop,\n"
        repr += "  transform = %s,\n" % str(self.global_rrc) + str(self.local_rrc) + str(self.random_flip) + str(self.format_transform)
        repr += ")"
        return repr


class DualTransform(object):
    def __init__(self, global_size=224, global_scale=(0.3, 1.0), post_trans=None, aug_trans=None, aug_trans_param=None):
        self.rrc_trans = SingleRandomResizedCrop(global_size, scale=global_scale, interpolation=3)
        self.aug_trans = aug_trans
        self.aug_trans_param = aug_trans_param
        self.random_flip = RandomHorizontalFlip()
        self.format_transform = post_trans
        logging.info(f'Dual Parameterized Crop with {global_size}@{global_scale}')

    def __call__(self, image):
        image = self.aug_trans(image) # color transform on original image
        
        image1, flip1 = self.random_flip(image)
        image1, i1, j1, h1, w1, W = self.rrc_trans(image1)
        image_aug1, aug_params1 = self.aug_trans_param(image1, aug_only=True)

        image2, flip2 = self.random_flip(image)
        image2, i2, j2, h2, w2, W = self.rrc_trans(image2)
        image_aug2, aug_params2 = self.aug_trans_param(image2, aug_only=True)
        
        relative_flip = float((flip1 and not flip2) or (flip2 and not flip1))
        # predict image2 from image1
        rel_pos_21 = torch.tensor([(i2-i1)/h1, (j2-j1)/w1, h2/h1, w2/w1, relative_flip, (W-j1-j2)/w1], dtype=torch.float)
        rel_pos_12 = torch.tensor([(i1-i2)/h2, (j1-j2)/w2, h1/h2, w1/w2, relative_flip, (W-j2-j1)/w2], dtype=torch.float)
        
        image1, image_aug1 = self.format_transform(image1), self.format_transform(image_aug1)
        image2, image_aug2 = self.format_transform(image2), self.format_transform(image_aug2)

        return image1, image_aug1, aug_params1, image2, image_aug2, aug_params2, rel_pos_21, rel_pos_12

    def __repr__(self):
        repr = "(ParameterizedCrop,\n"
        repr += "  transform = %s,\n" % str(self.rrc_trans) + str(self.aug_trans) + str(self.random_flip) + str(self.format_transform)
        repr += ")"
        return repr


class DualTransformEasy(object):
    def __init__(self, global_size=224, global_scale=(0.3, 1.0), post_trans=None, aug_trans=None, aug_trans_param=None, min_overlap=-1, flip_prob=0.5):
        self.rrc_trans = SingleRandomResizedCrop(global_size, scale=global_scale, interpolation=3)
        self.aug_trans = aug_trans
        self.aug_trans_param = aug_trans_param
        if flip_prob < 0:
            # flip before
            logging.info('Flip Before!!')
            self.random_flip = RandomHorizontalFlip(p=0.5)
            self.flip_before = True
        else:
            logging.info('Flip After!!')
            self.random_flip = RandomHorizontalFlip(p=flip_prob)
            self.flip_before = False
        self.format_transform = post_trans
        self.min_overlap = min_overlap
        logging.info(f'Dual Easy Parameterized Crop with {global_size}@{global_scale}, min_overlap={min_overlap}, flip={flip_prob}')
    
    
    def __call__(self, image):
        image = self.aug_trans(image) # color transform on original image
        if self.flip_before:
            image, _ = self.random_flip(image)
        width, height = F.get_image_size(image)
        
        if self.flip_before:
            image1, flip1 = image, False
        else:
            image1, flip1 = self.random_flip(image)
        image1, i1, j1, h1, w1, W = self.rrc_trans(image1)
        image_aug1, aug_params1 = self.aug_trans_param(image1, aug_only=True)

        if self.min_overlap >= 0:
            i_min = max(0, round(i1 - (1 - self.min_overlap) * h1))
            i_max = min(height - h1 + 1, round(i1 + (1 - self.min_overlap) * h1)+1)
            j_min = max(0, round(j1 - (1 - self.min_overlap) * w1))
            j_max = min(width - w1 + 1, round(j1 + (1 - self.min_overlap) * w1)+1)
            i2 = torch.randint(i_min, i_max, size=(1,)).item()  # same size crop
            j2 = torch.randint(j_min, j_max, size=(1,)).item()
        else:
            # image2, i2, j2, h2, w2, W = self.rrc_trans(image2)
            i2 = torch.randint(0, height - h1 + 1, size=(1,)).item() # same size crop
            j2 = torch.randint(0, width - w1 + 1, size=(1,)).item()

        if self.flip_before:
            image2, flip2 = image, False
        else:
            image2, flip2 = self.random_flip(image)
        image2 = F.resized_crop(image2, i2, j2, h1, w1, self.rrc_trans.size, self.rrc_trans.interpolation)
        image_aug2, aug_params2 = self.aug_trans_param(image2, aug_only=True)
        
        
        relative_flip = float((flip1 and not flip2) or (flip2 and not flip1))
        # predict image2 from image1
        rel_pos_21 = torch.tensor([(i2-i1)/h1, (j2-j1)/w1, relative_flip, (W-j1-j2)/w1], dtype=torch.float)
        rel_pos_12 = torch.tensor([(i1-i2)/h1, (j1-j2)/w1, relative_flip, (W-j2-j1)/w1], dtype=torch.float)
        
        image1, image_aug1 = self.format_transform(image1), self.format_transform(image_aug1)
        image2, image_aug2 = self.format_transform(image2), self.format_transform(image_aug2)

        return image1, image_aug1, aug_params1, image2, image_aug2, aug_params2, rel_pos_21, rel_pos_12

    def __repr__(self):
        repr = "(ParameterizedCrop,\n"
        repr += "  transform = %s,\n" % str(self.rrc_trans) + str(self.aug_trans) + str(self.random_flip) + str(self.format_transform)
        repr += ")"
        return repr



def build_world_model_dual_transform(args):
    mean, std = get_mean_std(args)
    post_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    aug_trans = transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=args.iwm_jitter_prob)
    aug_trans_param = build_parameterized_aug(args, post_trans)
    args.policy_dim = aug_trans_param.action_dim
    trans_cls = DualTransformEasy if 'easy' in args.ssl_type else DualTransform
    logging.info(f'Dual Transform Class: {trans_cls}')
    return trans_cls(
        global_scale=args.extra_global_scale, 
        post_trans=post_trans,
        aug_trans=aug_trans,
        aug_trans_param=aug_trans_param,
        min_overlap=args.min_overlap,
        flip_prob=args.flip_prob
    )
