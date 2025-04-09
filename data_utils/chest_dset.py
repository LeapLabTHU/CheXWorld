
import torch.utils.data as data
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import torch
import numpy as np
from .data_path import get_dataset_path


CHEST_DEFAULT_PATH = get_dataset_path('MedFMC_chest_512')

class MedFMC_Chest(data.Dataset):
    def __init__(self, root=CHEST_DEFAULT_PATH, exp_num=1, shots=10, transform=None, split='train', preload=True):
        super().__init__()
        self.root = root
        self.classes = ['pleural_effusion', 'nodule','pneumonia','cardiomegaly', 'hilar_enlargement', 'fracture_old', 'fibrosis',	'aortic_calcification', 'tortuous_aorta', 'thickened_pleura', 'TB', 'pneumothorax', 'emphysema', 'atelectasis', 'calcification','pulmonary_edema', 'increased_lung_markings', 'elevated_diaphragm', 'consolidation']
        
        self.num_classes = len(self.classes)
        
        assert split in ('train', 'val', 'test')
        filename = 'test.txt' if split == 'test' else f'chest_{shots}-shot_{split}_exp{exp_num}.txt'
        with open(os.path.join(f'annotations/medfmc/{filename}'), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip() != '']
            self.data_list = [line.split(' ')[0].replace('.png', '.jpg') for line in lines]
            self.label_list = [line.split(' ')[1].split(',') for line in lines]
            self.label_list = [[int(l) for l in lab] for lab in self.label_list]
        self.label_list = np.array(self.label_list, dtype=float)
        self.image_list = []
        for img_path in self.data_list:
            img_path = os.path.join(self.root, img_path)
            if preload:
                img = Image.open(img_path).convert('RGB')
            else:
                img = img_path
            self.image_list.append(img)
        self.preload = preload
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img, label = self.image_list[index], self.label_list[index]
        if not self.preload:
            img = Image.open(img).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, torch.FloatTensor(label)



class ShenZhenCXR(data.Dataset):
    def __init__(self, root=get_dataset_path('ShenZhen'), split='train', transform=None, preload=True):
        super().__init__()
        self.img_list = []
        self.img_label = []
        self.transform = transform
        self.num_classes = 1
        file_path = f'annotations/shenzhen/ShenzhenCXR_{split}_data.txt'
        self.root = root
        with open(file_path, "r") as fileDescriptor:
            line = True

            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split(',')

                    imagePath = os.path.join(root, lineItems[0])
                    imageLabel = lineItems[1:self.num_classes + 1]
                    imageLabel = [int(i) for i in imageLabel]

                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)


    def __getitem__(self, index):
        img, label = self.img_list[index], self.img_label[index]
        img = Image.open(img).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        return img, torch.FloatTensor(label)

    def __len__(self):
        return len(self.img_list)
