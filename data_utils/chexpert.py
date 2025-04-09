import torch.utils.data as data
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import torch
import logging
from .data_path import get_dataset_path

CHEXPERT_IMBALANCE_RATE = [2.20, 7.17, 13.71, 2.11, 1.21]
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}
CHEXPERT_COMPETITION_TASKS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

class CheXpert(data.Dataset):
    def __init__(self, transform=None, return_label=False, data_pct=1.0, split='train', enhance_class=['Cardiomegaly', 'Consolidation'], enhance_times=0, include_lateral=False):
        super().__init__()
        root = get_dataset_path('CheXpert')
        self.CLASS_NAMES = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'pleural effusion']
        assert split in ('train', 'val', 'test')
        SPLIT_MAP = {'train': 'train', 'val': 'valid', 'test': 'valid'}
        self.pathologies = CHEXPERT_COMPETITION_TASKS
        self.num_classes = len(self.pathologies)
        self.df = pd.read_csv(os.path.join(root, f'{SPLIT_MAP[split]}.csv'))
        if not include_lateral:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        if split == 'train':
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.split = split
        # # fill na with 0s
        # self.df = self.df.fillna(0)
        # # replace uncertains
        # uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        # self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
        
        for col in CHEXPERT_COMPETITION_TASKS: # from libauc
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)  
                self.df[col].fillna(0, inplace=True) 
            elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            elif col in ['No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other','Fracture','Support Devices']: # other labels
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self.labels = self.df[self.pathologies].values.tolist()
        self.img_list = [os.path.join(root, path) for path in self.df['Path'].tolist()]
        
        if split == 'train':
            en_img_list = []
            en_labels = []
            en_indices = [CHEXPERT_COMPETITION_TASKS.index(en_cls) for en_cls in enhance_class]
            for img, lab in zip(self.img_list, self.labels):
                en = False
                for en_index in en_indices:
                    if lab[en_index] == 1:
                        en = True
                if en:
                    en_img_list += [img] * enhance_times
                    en_labels += [lab] * enhance_times
            
            self.img_list.extend(en_img_list)
            self.labels.extend(en_labels)
    
        self.transform = transform
        self.return_label = return_label
        logging.info(f'CheXpert {split} size: {len(self.img_list)}')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path, label = self.img_list[index], self.labels[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform != None:
            img = self.transform(img)
        
        if not self.return_label:
            return img
        
        if self.split == 'train':
            return img, torch.FloatTensor(label)
        else:
            return img, img_path, torch.FloatTensor(label)



