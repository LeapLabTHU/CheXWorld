import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
import torch
from .data_path import get_dataset_path

# skip images from https://github.com/philip-mueller/adpd/blob/main/src/dataset/mimic_cxr_datasets.py
SKIP_IMAGES = [
    "0518c887-b80608ca-830de2d5-89acf0e2-bd3ec900",
    "03b2e67c-70631ff8-685825fb-6c989456-621ca64d",
    "786d69d0-08d16a2c-dd260165-682e66e9-acf7e942",
    "1d0bafd0-72c92e4c-addb1c57-40008638-b9ec8584",
    "f55a5fe2-395fc452-4e6b63d9-3341534a-ebb882d5",
    "14a5423b-9989fc33-123ce6f1-4cc7ca9a-9a3d2179",
    "9c42d877-dfa63a03-a1f2eb8c-127c60c3-b20b7e01",
    "996fb121-fab58dd2-7521fd7e-f9f3133c-bc202556",
    "56b8afd3-5f6d4419-8699d79e-6913a2bd-35a08557",
    "93020995-6b84ca33-2e41e00d-5d6e3bee-87cfe5c6",
    "f57b4a53-5fecd631-2fe14e8a-f4780ee0-b8471007",
    "d496943d-153ec9a5-c6dfe4c0-4fb9e57f-675596eb",
    "46b02f13-69fb7e49-321880e4-80584065-c1f57b50",
    "422689b1-40e06ae8-d6151ff3-2780c186-6bd67271",
    "8385a8ad-ad5e02a8-8e1fa7f3-d822c648-2a41a205",
    "e180a7b6-684946d6-fe1782de-45ed1033-1a6f8a51",
    "f5f82c2f-e99a7a06-6ecc9991-072adb2f-497dae52",
    "6d54a492-7aade003-a238dc5c-019ccdd2-05661649",
    "2b5edbbf-116df0e3-d0fea755-fabd7b85-cbb19d84",
    "db9511e3-ee0359ab-489c3556-4a9b2277-c0bf0369",
    "87495016-a6efd89e-a3697ec7-89a81d53-627a2e13",
    "810a8e3b-2cf85e71-7ed0b3d3-531b6b68-24a5ca89",
    "a9f0620b-6e256cbd-a7f66357-2fe78c8a-49caac26",
    "46b02f13-69fb7e49-321880e4-80584065-c1f57b50",
]
class MIMIC_Pretrain(Dataset):
    def __init__(self, path=None, transform=None, split='train', data_pct=1.0, preload=False, data_postfix='', include_lateral=False):
        super().__init__()
        path = get_dataset_path('MIMIC' + data_postfix)
        self.data_root = path
        self.imgpath = os.path.join(path, "files_512")
        logging.info(f'MIMIC from {path}')
        def extract_imgpath(row):
            return str(os.path.join(
                self.imgpath, 
                "p" + str(row['subject_id'])[:2], 
                "p" + str(row['subject_id']),
                "s" + str(row['study_id']), 
                str(row['dicom_id']) + ".jpg"
            ))
        self.csv = pd.read_csv(os.path.join(path, "mimic-cxr-2.0.0-chexpert.csv.gz"))
        self.metacsv = pd.read_csv(os.path.join(path, "mimic-cxr-2.0.0-metadata.csv.gz"))
        self.splitcsv = pd.read_csv(os.path.join(path, 'mimic-cxr-2.0.0-split.csv.gz'))
        views = ["AP", "PA"]
        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        self.metacsv = self.metacsv.set_index(['subject_id', 'study_id'])
        self.csv = self.csv.join(self.metacsv).reset_index()
        self.csv = self.csv.set_index(['subject_id', 'study_id', 'dicom_id'])
        self.splitcsv = self.splitcsv.set_index(['subject_id', 'study_id', 'dicom_id'])
        self.csv = self.csv.join(self.splitcsv).reset_index()
        self.csv['imgpath'] = self.csv.apply(extract_imgpath, axis=1)
        
        if not include_lateral:
            self.csv = self.csv[self.csv["ViewPosition"].isin(views)]
        self.split = split
        assert split in ('all', 'train', 'validate', 'test')
        if split != 'all':
            self.csv = self.csv[self.csv['split'] == 'train']
        if split in ['all', 'train'] and data_pct < 1.0:
            self.csv = self.csv.sample(frac=data_pct, random_state=42)
        self.csv = self.csv[~self.csv['dicom_id'].isin(SKIP_IMAGES)]
        self.transform = transform
        self.preload = preload

    def __len__(self):
        return len(self.csv.index)

    def __getitem__(self, idx):
        if self.preload:
            img  = self.images[idx].convert('RGB')
        else:
            img_path = self.csv.iloc[idx]['imgpath']
            img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

