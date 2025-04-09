from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image, ImageFilter, ImageOps
import logging
import numpy as np
import torch
from timm.data import create_transform
import random
from .randaugment import RandomAugment

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class RandomGaussianNoise:
    def __init__(self, noise_range=(0.05, 0.2)):
        self.noise_range = noise_range
    
    def __call__(self, img):
        sig = float(torch.empty(1).uniform_(self.noise_range[0], self.noise_range[1]))
        return add_gaussian_noise_pil(img, sig)

def add_gaussian_noise_pil(img, sig):
    arr = np.asarray(img)
    arr1 = arr + np.random.randn(*arr.shape) * sig * 255
    arr1 = np.clip(arr1, 0, 255)
    arr1 = arr1.astype(arr.dtype)
    img = Image.fromarray(arr1)
    return img


XRAY_MEAN = [0.4978]
XRAY_STD = [0.2449]

def get_mean_std(args):
    if args.norm_type == 'default':
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif args.norm_type == 'inception':
        mean = IMAGENET_INCEPTION_MEAN
        std = IMAGENET_INCEPTION_STD
    elif args.norm_type == 'xray': # from medmae
        mean = [0.5056, 0.5056, 0.5056]
        std = [0.252, 0.252, 0.252]
    else:
        raise NotImplementedError
    return mean, std

class Identity:
    def __call__(self, x):
        return x


def build_aug_trans(args):
    aug_trans = []
    if args.rot > 0:
        logging.info(f'Rotation={args.rot}')
        aug_trans.append(transforms.RandomRotation(degrees=args.rot, interpolation=3))
    has_blur = 'blur' in args.aug_type
    has_sol = 'sol' in args.aug_type
    if has_blur or has_sol:
        logging.info(f'blur {has_blur}, sol {has_sol}, p={args.iwm_blur_prob}')
        if has_blur and has_sol:
            t = transforms.RandomChoice([Solarization(p=1.0), GaussianBlur(p=1.0)])
        elif has_blur:
            t = GaussianBlur(p=1.0)
        else:
            t = Solarization(p=1.0) 
        aug_trans.append(transforms.RandomApply([t], p=args.iwm_blur_prob))
    if 'jit' in args.aug_type:
        logging.info(f'Jitter={args.color_jitter}')
        aug_trans.append(transforms.RandomApply([transforms.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter)], p=1.0))
    if 'noise' in args.aug_type:
        aug_trans.append(transforms.RandomApply([RandomGaussianNoise(noise_range=args.iwm_noise_range)], p=args.iwm_noise_prob))
    if 'ra' in args.aug_type:
        logging.info(f'Random Augment (2,7)')
        trans = RandomAugment(2,7, isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'])
        aug_trans.append(trans)
    aug_trans = transforms.Compose(aug_trans)
    return aug_trans



def build_transform_crop(args, is_train=True):
    img_size = args.input_size
    resize_size = args.input_size * 512 // 448 if args.resize_size is None else args.resize_size
    logging.info(f'resize size: {args.resize_size}, input size: {args.input_size}')
    if is_train:
        if args.crop_type == 'rc':
            crop_trans = transforms.Compose([
                transforms.Resize(resize_size, interpolation=Image.BICUBIC),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
            ])
        elif args.crop_type == 'rrc':
            logging.info(f'Scale Min: {args.scale_min}')
            crop_trans = transforms.Compose([
                transforms.RandomResizedCrop(img_size,scale=(args.scale_min, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            raise NotImplementedError
    else:
        crop_trans = transforms.Compose([
            transforms.Resize(resize_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop((img_size, img_size)),
        ])
    return crop_trans


def build_transform_xray(args, is_train, ten_crop=False):
    mean, std = get_mean_std(args)
    post_trans = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    if args.aug_type == 'timm' and is_train:
        logging.info(f'Use timm transform with aa={args.aa}, scale_min={args.scale_min}')
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            scale=(args.scale_min, 1.0),
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=0,
            mean=mean,
            std=std,
        )
        return transform
    
    logging.info(f'crop type: {args.crop_type}, aug type: {args.aug_type}')
    crop_trans = build_transform_crop(args, is_train)
    if is_train:
        aug_trans = build_aug_trans(args)
    else:
        aug_trans = Identity()

    if ten_crop:
        logging.info('Build TenCrop Transform!')
        transform = transforms.Compose([
            transforms.Resize(args.resize_size, interpolation=Image.BICUBIC),
            transforms.TenCrop(args.input_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([post_trans(crop) for crop in crops])) # returns a 4D tensor
        ])
    else:
        transform = transforms.Compose([crop_trans, aug_trans, post_trans])
    return transform




def build_simclr_transform(args, img_size=224):
    mean, std = get_mean_std(args)
    transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=(0.2, 0.8)),
            transforms.RandomAffine(degrees=(-20, 20)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2))], p=0.5),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
    ])
    return transform


import albumentations
from albumentations.pytorch.transforms import ToTensorV2

def build_seg_transform(args, is_train):
    mean, std = get_mean_std(args)
    if is_train:
        if args.aug_type == 'adam':
            data_transforms = albumentations.Compose([
                albumentations.Resize(224, 224),
                albumentations.OneOf([
                    albumentations.RandomBrightnessContrast(),
                    albumentations.RandomGamma(),
                    ], p=0.3),
                albumentations.OneOf([
                    albumentations.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    albumentations.GridDistortion(),
                    albumentations.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ], p=0.3),
                albumentations.RandomSizedCrop(min_max_height=(156, 224), height=224, width=224,p=0.25),
                albumentations.Normalize(mean, std),
                ToTensorV2()
            ],p=1)
        else:
            jit_aug = albumentations.RandomBrightnessContrast(
                                            brightness_limit=args.color_jitter,
                                            contrast_limit=args.color_jitter,
                                            p=args.iwm_jitter_prob
                                            )
            rot_aug = albumentations.ShiftScaleRotate(rotate_limit=args.rot, border_mode=0)
            logging.info(f'Scale Aug Type: {args.aug_type}')
            logging.info(f'Jitter: {args.color_jitter} (p={args.iwm_jitter_prob}), Rot: {args.rot}')
            data_transforms = albumentations.Compose([
                albumentations.Resize(args.input_size, args.input_size),
                rot_aug,
                jit_aug,
                albumentations.HorizontalFlip(p=0.5),
                albumentations.Normalize(mean, std),
                ToTensorV2()
                ])
    else:
        data_transforms = albumentations.Compose([
                                            albumentations.Resize(args.input_size, args.input_size),
                                            albumentations.Normalize(mean, std),
                                            ToTensorV2()
                                            ])
    return data_transforms