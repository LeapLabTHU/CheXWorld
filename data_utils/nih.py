import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import logging
from .data_path import get_dataset_path


class NIH(data.Dataset):
    pathologies = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    def __init__(self, split="train", data_pct=1.0, transform=None, return_label=False):
        super().__init__()
        self.root = get_dataset_path('NIH')
        self.return_label = return_label
        self.num_classes = len(self.pathologies)
        self.split = split
        self.transform = transform
        # if os.path.exists(os.path.join(self.root, 'NIH_proc')):
        #     logging.info('NIH Proc')
        #     self.nih_proc = True
        # else:
        #     logging.info('NIH Original')
        #     self.nih_proc = False
        self.nih_proc = False
        if not return_label:
            logging.info('NIH for Pretrain on train_val')
            with open(os.path.join(self.root, 'train_val_list.txt'), 'r') as f:
                lines = f.readlines()
                if self.nih_proc:
                    self.data = [os.path.join(self.root, 'NIH_proc', l.strip()) for l in lines if l.strip() != '']
                else:
                    self.data = [os.path.join(self.root, 'NIH_512_jpg', l.strip()) for l in lines if l.strip() != '']
            logging.info(f'NIH Num Pretrain: {len(self.data)}')
        else:
            assert split in ('train', 'val', 'test')
            self.data = []
            self.img_label = []
            img_root = os.path.join(self.root, 'NIH_512')
            if not os.path.exists(img_root):
                img_root = img_root.replace('NIH_512', 'NIH_512_jpg')
            with open(f'annotations/nih/Xray14_{split}_official.txt', "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline()
                    if line:
                        lineItems = line.split()
                        imagePath = os.path.join(img_root, lineItems[0])

                        imageLabel = lineItems[1:self.num_classes + 1]
                        imageLabel = [int(i) for i in imageLabel]

                        self.data.append(imagePath)
                        self.img_label.append(imageLabel)
                if data_pct < 1.0 and split == 'train':
                    state = np.random.RandomState(seed=42)
                    num_data = len(self.data)
                    indices = state.choice(np.arange(num_data), size=int(data_pct * num_data), replace=False)
                    self.data = [self.data[idx] for idx in indices]
                    self.img_label = [self.img_label[idx] for idx in indices]
            logging.info(f'NIH Downstream {split} set: {len(self.data)}')

    def __getitem__(self, index):
        if not self.return_label:
            img_path = self.data[index]
        else:
            img_path = self.data[index]
            label = self.img_label[index]
        
        # if self.nih_proc:
        #     img = Image.open(img_path.replace('.png', '.jpg')).convert('RGB')
        # else:
        #     img = Image.open(img_path).convert('RGB')
        if 'jpg' in img_path:
            img = Image.open(img_path.replace('.png', '.jpg')).convert('RGB')
        else:
            img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        
        if not self.return_label:
            return img
        else:
            return img, torch.FloatTensor(label)

    def __len__(self):
        return len(self.data)
