import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch
import numbers
import math
from PIL import ImageFilter, Image
import logging
import numpy as np
from .xray_transform import get_mean_std, Identity, add_gaussian_noise_pil, build_transform_crop


def normalize_factor(factor, factor_range):
    min, max = factor_range
    f01 = (factor - min) / (max - min)
    return f01 * 2 - 1

class ParameterizedTransform:
    ''' Assume the images are PIL'''
    def __init__(self, 
                 jitter_prob=0.0, brightness=0.0, contrast=0.0, gamma=(0.5, 2.0),
                 blur_prob=0.0, blur_radius=(0.1, 2.0), noise_prob=0.0, noise_range=(0.05, 0.2), post_trans=None, aug_norm=False):
        if isinstance(brightness, numbers.Number):
            brightness = (max(1-brightness, 0), 1+brightness)
        if isinstance(contrast, numbers.Number):
            contrast = (max(1-contrast, 0), 1+contrast)
        if isinstance(gamma, numbers.Number):
            gamma = (min(1/gamma, gamma), max(1/gamma, gamma))
        self.jitter_prob, self.brightness, self.contrast, self.gamma = jitter_prob, brightness, contrast, gamma
        self.blur_prob, self.blur_radius = blur_prob, blur_radius
        self.noise_prob, self.noise_range = noise_prob, noise_range
        # self.solarize_prob = self.solarize_prob
        self.jitter_dim = 3
        self.blur_dim = 1
        self.noise_dim = 1
        self.action_dim = self.jitter_dim + self.blur_dim + self.noise_dim
        self.post_trans = post_trans
        self.aug_norm = aug_norm
        logging.info(
            f'Transform Config:\njitter p={jitter_prob} brightness={self.brightness} contrast={self.contrast} gamma={self.gamma}\nblur p={blur_prob} radius={blur_radius}\nnoise p={noise_prob} range={noise_range} aug_norm={aug_norm}'
        )
    def apply_jitter(self, img):
        if self.jitter_prob < torch.rand(1):
            brightness_factor, contrast_factor, gamma_factor = 1.0, 1.0, 1.0
        else:
            brightness_factor = float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
            contrast_factor = float(torch.empty(1).uniform_(self.contrast[0], self.contrast[1]))
            gamma_factor = math.exp(float(torch.empty(1).uniform_(math.log(self.gamma[0]), math.log(self.gamma[1])))) # sample gamma in log space
        img = F.adjust_brightness(img, brightness_factor)
        img = F.adjust_contrast(img, contrast_factor)
        img = F.adjust_gamma(img, gamma_factor)
        
        if self.aug_norm:
            brightness_factor = normalize_factor(brightness_factor, self.brightness)
            contrast_factor = normalize_factor(contrast_factor, self.contrast)
            gamma_factor = normalize_factor(math.log(gamma_factor), (math.log(self.gamma[0]), math.log(self.gamma[1])))
        return img, torch.tensor([brightness_factor, contrast_factor, gamma_factor])
    
    def apply_blur(self, img):
        if self.blur_prob < torch.rand(1):
            radius = 0
        else:
            radius = float(torch.empty(1).uniform_(self.blur_radius[0], self.blur_radius[1]))
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        if self.aug_norm:
            radius = radius / self.blur_radius[1] # normalized to [0, 1]
        return img, torch.tensor([radius])

    def apply_noise(self, img):
        if self.noise_prob < torch.rand(1):
            sig = 0
        else:
            sig = float(torch.empty(1).uniform_(self.noise_range[0], self.noise_range[1]))
            img = add_gaussian_noise_pil(img, sig)
        return img, torch.tensor([sig])

    def __call__(self, img, aug_only=False):
        ori_img = img
        img, jitter_params = self.apply_jitter(img)
        img, blur_params = self.apply_blur(img)
        img, noise_params = self.apply_noise(img)
        aug_params = torch.cat([jitter_params, blur_params, noise_params])
        if aug_only:
            return img, aug_params
        if self.post_trans is not None:
            img, ori_img = self.post_trans(img), self.post_trans(ori_img)
        return img, ori_img, aug_params



def build_parameterized_aug(args, post_trans):
    if args.iwm_version == 'v1':
        iwm_trans = ParameterizedTransform(
        jitter_prob=args.iwm_jitter_prob,brightness=0.4,contrast=0.4,gamma=(0.5,2.0),
        blur_prob=args.iwm_blur_prob, 
        noise_prob=args.iwm_noise_prob, 
        post_trans=post_trans,
        aug_norm=args.iwm_aug_norm
    )
    else:
        raise NotImplementedError
    logging.info(f'IWM Version: {args.iwm_version}')
    return iwm_trans

def build_world_model_transform(args):
    img_size = args.input_size
    resize_size = args.input_size * 512 // 448 if args.resize_size is None else args.resize_size
    logging.info(f'resize size: {args.resize_size}, input size: {args.input_size}')
    mean, std = get_mean_std(args)
    post_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    logging.info(f'crop type: {args.crop_type}, aug type: {args.aug_type}')
    crop_trans = build_transform_crop(args)

    if args.aug_type == 'jit':
        aug_trans = transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=args.iwm_jitter_prob) # follow IWM
    elif args.aug_type == 'none':
        aug_trans = Identity()
    else:
        raise NotImplementedError

    iwm_trans = build_parameterized_aug(args, post_trans)
    args.policy_dim = iwm_trans.action_dim
    transform = transforms.Compose([crop_trans, aug_trans, iwm_trans])
    return transform