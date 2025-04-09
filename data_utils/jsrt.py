from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from PIL import Image
from .data_path import get_dataset_path


## JSRT downloaded from: https://github.com/AnishHota/JSRT-Segmentation
# One fold contained all 124 odd numbered images in the JSRT database. The other fold contained the 123 even numbered images. 
class JSRT(Dataset):
    def __init__(self, split='train', transform=None, mode='heart', dataset_seed=42):
        super().__init__()
        self.split = split
        self.root = get_dataset_path('JSRT')
        self.transform = transform
        self.mode = mode
        train_list, test_list = [], []
        for p in os.listdir(os.path.join(self.root, 'images')):
            ind = int(p.replace('JPCLN', '').replace('JPCNN', '').split('.')[0])
            if ind % 2 != 0:
                train_list.append(p)
            else:
                test_list.append(p)
        train_list, val_list = train_test_split(train_list, train_size=0.9, random_state=dataset_seed)
        logging.info(f'JSRT Dataset Seed: {dataset_seed}')
        self.data_list = dict(train=train_list, val=val_list, test=test_list)[split]
        self.num_classes = 1
        self.mode = mode
        assert mode in ('heart', 'lung', 'clavicle')
        logging.info(f'JSRT Mode={mode}, split={split}, #data={len(self.data_list)}')

    def __getitem__(self, index):
        p = self.data_list[index]
        image = cv2.imread(os.path.join(self.root, f'images', p))
        image = cv2.resize(image, (1024, 1024))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mode == 'heart':
            mask = cv2.imread(os.path.join(self.root, 'masks', 'heart', p.replace('.jpg', '.png')))[:, :, 1]
            mask[mask > 0]=1
        elif self.mode == 'clavicle':
            mask1 = cv2.imread(os.path.join(self.root, 'masks', 'left_clavicle', p.replace('.jpg', '.png')))[:, :, 1]
            mask1[mask1 > 0]=1
            mask2 = cv2.imread(os.path.join(self.root, 'masks', 'right_clavicle', p.replace('.jpg', '.png')))[:, :, 1]
            mask2[mask2 > 0]=1
            mask = np.logical_or(mask1, mask2).astype(int)
        elif self.mode == 'lung':
            mask1 = cv2.imread(os.path.join(self.root, 'masks', 'left_lung', p.replace('.jpg', '.png')))[:, :, 1]
            mask1[mask1 > 0]=1
            mask2 = cv2.imread(os.path.join(self.root, 'masks', 'right_lung', p.replace('.jpg', '.png')))[:, :, 1]
            mask2[mask2 > 0]=1
            mask = np.logical_or(mask1, mask2).astype(int)
        else:
            raise NotImplementedError
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)
        return image.float(), mask.float()

    def __len__(self):
        return len(self.data_list)







import os 

import torch
from skimage import io, transform

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from skimage.transform import resize
class JSRT_Landmark(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.root = get_dataset_path('JSRT_jpg')
        train_list, test_list = [], []
        landmark_path = 'annotations/landmarks_jsrt'
        train_landmarks = []
        test_landmarks = []
        for p in os.listdir(landmark_path + '/H'):
            ind = int(p.replace('JPCLN', '').replace('JPCNN', '').split('.')[0])
            h = np.load(os.path.join(landmark_path, 'H', p)).astype('float')
            ll = np.load(os.path.join(landmark_path, 'LL', p)).astype(np.float32)
            rl = np.load(os.path.join(landmark_path, 'RL', p)).astype(np.float32)
            landmarks = np.concatenate([rl, ll, h], axis = 0)
            if ind % 2 != 0:
                train_list.append(p.replace('.npy', '.jpg'))
                train_landmarks.append(landmarks)
            else:
                test_list.append(p.replace('.npy', '.jpg'))
                test_landmarks.append(landmarks)

        self.data_list = dict(train=train_list, val=test_list, test=test_list)[split]
        self.landmarks = dict(train=train_landmarks, val=test_landmarks, test=test_landmarks)[split]
            
        self.num_classes = 240

        if split == 'train':
            self.transform = transforms.Compose([
                                                #  RandomScale(),
                                                #  Rotate(3),
                                                #  AugColor(0.40),
                                                 ToTensor(),
                                                 ])
        else:
            self.transform = ToTensor()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_name = self.data_list[idx]
        img_path = os.path.join(self.root, img_name)
                         
        image = io.imread(img_path).astype('float') / 255.0
        # image = resize(image, output_shape=(1024, 1024, 3))
        # image = np.expand_dims(image, axis=2)
        # print(image.shape)
        landmarks = self.landmarks[idx].copy()
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['landmarks']

    

    
class RandomScale(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']       
        
        # Pongo limites para evitar que los landmarks salgan del contorno
        min_x = np.min(landmarks[:,0]) 
        max_x = np.max(landmarks[:,0])
        ancho = max_x - min_x
        
        min_y = np.min(landmarks[:,1])
        max_y = np.max(landmarks[:,1])
        alto = max_y - min_y
        
        max_var_x = 1024 / ancho 
        max_var_y = 1024 / alto
        min_var_x = 0.80
        min_var_y = 0.80
                                
        varx = np.random.uniform(min_var_x, max_var_x)
        vary = np.random.uniform(min_var_y, max_var_y)
                
        landmarks[:,0] = landmarks[:,0] * varx
        landmarks[:,1] = landmarks[:,1] * vary
        
        h, w = image.shape[:2]
        new_h = np.round(h * vary).astype('int')
        new_w = np.round(w * varx).astype('int')

        img = transform.resize(image, (new_h, new_w))
        
        # Cropeo o padeo aleatoriamente
        min_x = np.round(np.min(landmarks[:,0])).astype('int')
        max_x = np.round(np.max(landmarks[:,0])).astype('int')
        
        min_y = np.round(np.min(landmarks[:,1])).astype('int')
        max_y = np.round(np.max(landmarks[:,1])).astype('int')
        
        if new_h > 1024:
            rango = 1024 - (max_y - min_y)
            maxl0y = new_h - 1025
            
            if rango > 0 and min_y > 0:
                l0y = min_y - np.random.randint(0, min(rango, min_y))
                l0y = min(maxl0y, l0y)
            else:
                l0y = min_y
            l1y = l0y + 1024
            
            img = img[l0y:l1y,:]
            landmarks[:,1] -= l0y
            
        elif new_h < 1024:
            pad = h - new_h
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((p0, p1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,1] += p0
        
        if new_w > 1024:
            rango = 1024 - (max_x - min_x)
            maxl0x = new_w - 1025
            
            if rango > 0 and min_x > 0:
                l0x = min_x - np.random.randint(0, min(rango, min_x))
                l0x = min(maxl0x, l0x)
            else:
                l0x = min_x
            
            l1x = l0x + 1024
                
            img = img[:, l0x:l1x]
            landmarks[:,0] -= l0x
            
        elif new_w < 1024:
            pad = w - new_w
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((0, 0), (p0, p1), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,0] += p0
        
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            print('Original', [new_h,new_w])
            print('Salida', img.shape)
            raise Exception('Error')
            
        return {'image': img, 'landmarks': landmarks}
    
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return np.float32(cv2.LUT(image.astype('uint8'), table))

class AugColor(object):
    def __init__(self, gammaFactor):
        self.gammaf = gammaFactor

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        # Gamma
        gamma = np.random.uniform(1 - self.gammaf, 1 + self.gammaf / 2)
        
        image[:,:,0] = adjust_gamma(image[:,:,0] * 255, gamma) / 255
        
        # Adds a little noise
        image = image + np.random.normal(0, 1/128, image.shape)
        
        return {'image': image, 'landmarks': landmarks}

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        angle = np.random.uniform(- self.angle, self.angle)

        image = transform.rotate(image, angle)
        
        centro = image.shape[0] / 2, image.shape[1] / 2
        
        landmarks -= centro
        
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        landmarks = np.dot(landmarks, R)
        
        landmarks += centro

        return {'image': image, 'landmarks': landmarks}

import torchvision.transforms.functional
import torch.nn.functional as F
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                
        size = image.shape[0]
        image = image.transpose((2, 0, 1))
        landmarks = landmarks / size
        landmarks = np.clip(landmarks, 0, 1)
        
        image = torch.from_numpy(image).float()
        image = transforms.functional.normalize(image, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        return {'image': image,
                'landmarks': torch.from_numpy(landmarks).float()}