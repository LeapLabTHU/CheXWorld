
import torch.utils.data as data
import os
from PIL import Image
import torch
import logging
import numpy as np
from .data_path import get_dataset_path
from sklearn.model_selection import train_test_split

VINDR_CLASSES = ['Lung Opacity', 'Cardiomegaly', 'Pleural thickening', 'Aortic enlargement', 'Pulmonary fibrosis', 'Tuberculosis',]


VIN_DEFAULT_PATH = get_dataset_path('vinbigdata')

class VinDrNew(data.Dataset):
    def __init__(self, root=VIN_DEFAULT_PATH, transform=None, data_pct=1.0, split='train', data_postfix='', return_label=True, dataset_seed=42):
        super().__init__()
        root = root + data_postfix
        logging.info(f'VinDr from {root}')
        assert split in ('train', 'val', 'test')
        self.classes = None
        self.num_classes = 6
        dataset_split = 'test' if split == 'test' else 'train'
        with open(f'annotations/vindr/VinDrCXR_{dataset_split}_pe_global_one.txt', 'r') as f:
            lines = f.readlines()
            self.img_list = []
            self.labels = []
            for l in lines:
                words = l.strip().split(' ')
                words = [w for w in words if w != '']
                self.img_list.append(os.path.join(root, 'images', words[0]+'.jpg'))
                self.labels.append([int(l) for l in words[1:]])
        
        if split != 'test':
            train_img_list, val_img_list, train_labels, val_labels = train_test_split(self.img_list, self.labels, train_size=0.9, random_state=dataset_seed)
            logging.info(f'Dataset Seed: {dataset_seed}')
            if split == 'train':
                self.img_list, self.labels = train_img_list, train_labels
            else:
                self.img_list, self.labels = val_img_list, val_labels
        
        if data_pct < 1.0 and split == 'train':
            state = np.random.RandomState(seed=dataset_seed)
            num_data = len(self.img_list)
            indices = state.choice(np.arange(num_data), size=int(data_pct * num_data), replace=False)
            self.img_list = [self.img_list[idx] for idx in indices]
            self.labels = [self.labels[idx] for idx in indices]
        self.labels = np.asarray(self.labels).astype(np.float32)
        logging.info(f'VinDr New {split} split: {len(self.img_list)} images')
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path, label = self.img_list[index], self.labels[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform != None:
            img = self.transform(img)

        if not self.return_label:
            return img
        else:
            return img, torch.FloatTensor(label)