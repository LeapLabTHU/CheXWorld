from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from .data_path import get_dataset_path

class SIIM(Dataset):
    def __init__(self, split='train', transform=None, data_pct=1.0, test_frac=0.15, label_mode='new', balance=0):
        super().__init__()
        self.split = split
        self.root = get_dataset_path('SIIM')
        self.transform = transform
        if label_mode == 'mrm':
            logging.info('SIIM MRM Split')
            df = pd.read_csv(os.path.join(self.root, 'stage_2_train.csv'))
            if split == 'train' and data_pct < 1.0:
                file_name = f'annotations/siim_mrm/train_10.txt'
            else:
                file_name = f'annotations/siim_mrm/{split}_list.txt'
            with open(file_name, 'r') as f:
                lines = f.readlines()
            self.img_ids = [l.strip() for l in lines]
        elif label_mode == 'new':
            file_name = f'annotations/siim_new/{split}.txt'
            with open(file_name, 'r') as f:
                lines = f.readlines()
            self.img_ids = [l.strip() for l in lines]
            if split == 'train' and data_pct < 1.0:
                _, self.img_ids = train_test_split(self.img_ids, test_size=data_pct , random_state=0)

        else:
            logging.info('SIIM Manual Split')
            df = pd.read_csv(os.path.join(self.root, 'stage_2_train.csv'))
            train_df, test_val_df = train_test_split(df, test_size=test_frac * 2, random_state=0)
            test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)
            self.df = {'train': train_df, 'val': valid_df, 'test': test_df}[split]
            self.img_ids = self.df['ImageId'].values

        if balance > 0 and split == 'train':
            pos_samples = []
            for img_id in self.img_ids:
                mask = cv2.imread(os.path.join(self.root, 'annotations_01', img_id+'.png'))
                if mask.sum() > 0:
                    pos_samples.append(img_id)
            logging.info(f'Balance = {balance} on {len(pos_samples)} Pos Samples')
            self.img_ids = self.img_ids + pos_samples * balance
        self.num_classes = 1
    

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root, f'images', self.img_ids[index]+'.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(os.path.join(self.root, 'annotations_01', self.img_ids[index]+'.png'))[:, :, 1]
        mask[mask > 0]=1
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)
        return image.float(), mask.float()

    def __len__(self):
        return len(self.img_ids)