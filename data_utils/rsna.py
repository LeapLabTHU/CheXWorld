import pydicom
import numpy as np
import os
from PIL import Image
import torch.utils.data as data
import torch
import pandas as pd
import logging
from .data_path import get_dataset_path

def dicom_to_pil_image(dicom_filename):
    # Load the DICOM image
    dicom_image = pydicom.dcmread(dicom_filename)

    # Get the image array from the DICOM file
    image_array = dicom_image.pixel_array

    # Normalize the image array to the range [0, 255] and convert to uint8
    image_array = np.uint8((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255)

    # Convert the normalized array to a PIL image
    pil_image = Image.fromarray(image_array)

    return pil_image

def create_bbox(row):
    if row["Target"] == 0:
        return 0
    else:
        x1 = row["x"]
        y1 = row["y"]
        x2 = x1 + row["width"]
        y2 = y1 + row["height"]
        return [int(x1), int(y1), int(x2), int(y2)]

class RSNA(data.Dataset):
    def __init__(self, root=get_dataset_path('RSNA'), split="train", data_pct=1.0, transform=None, return_label=True, mode='3_class'):
        super().__init__()
        self.imgpath = os.path.join(root, 'stage_2_train_images')
        assert split in ('train', 'val', 'test')
        
        df = pd.read_csv(os.path.join(root, 'stage_2_train_labels.csv'))
        df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

        # aggregate multiple boxes
        df = df[["patientId", "bbox"]]
        df = df.groupby("patientId").agg(list)
        df = df.reset_index()
        df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)
        # create labels
        df["Target"] = df["bbox"].apply(lambda x: 0 if x == None else 1)

        self.df = df
        # train_df, test_val_df = train_test_split(
        #     df, test_size=test_fac * 2, random_state=0)
        # test_df, valid_df = train_test_split(
        #     test_val_df, test_size=0.5, random_state=0)
        self.df["Path"] = self.df["patientId"].apply(
            lambda x: os.path.join(self.imgpath, x + ".jpg"))
        self.transform = transform
        self.CLASSES = ['Pneumonia']
        self.num_classes = 1
        if data_pct != 1 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.return_label = return_label
        logging.info(f'RSNA num data: {len(self.df)}')

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["Path"]
        # img = dicom_to_pil_image(img_path)
        img = Image.open(img_path).convert('RGB')
        y = float(row["Target"])
        y = torch.tensor([y])
        
        if self.transform != None:
            img = self.transform(img)
        
        if not self.return_label:
            return img
        else:
            return img, torch.FloatTensor(y)



class RSNA_3Class(data.Dataset):
    def __init__(self, root=get_dataset_path('RSNA'), split="train", data_pct=1.0, transform=None, return_label=True, mode='3_class'):
        super().__init__()
        # self.imgpath = os.path.join(root, )
        assert split in ('train', 'val', 'test')
        
        self.num_classes = 3
        with open(f'annotations/rsna/RSNAPneumonia_{split}.txt', 'r') as f:
            lines = f.readlines()
            self.img_list = []
            self.labels = []
            for l in lines:
                words = l.strip().split(' ')
                words = [w for w in words if w != '']
                if len(words) == 0: continue
                self.img_list.append(os.path.join(root, 'stage_2_train_images', words[0]+'.jpg'))
                label = [0.,0.,0.]
                label[int(words[-1])] = 1
                self.labels.append(label)
        if data_pct < 1.0 and split == 'train':
            state = np.random.RandomState(seed=42)
            num_data = len(self.img_list)
            indices = state.choice(np.arange(num_data), size=int(data_pct * num_data), replace=False)
            self.img_list = [self.img_list[idx] for idx in indices]
            self.labels = [self.labels[idx] for idx in indices]
        self.labels = np.asarray(self.labels).astype(np.float32)
        logging.info(f'RSNA_3Class {split} split: {len(self.img_list)} images')
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


import cv2

class RSNASegmentDataset:
    def __init__(self, root=get_dataset_path('RSNA'), split="train", data_pct=1.0, transform=None,) -> None:
        super().__init__()
        self.root = root
        self.split = split
        df = pd.read_csv(os.path.join(root, 'stage_2_train_labels.csv'))
        df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

        # aggregate multiple boxes
        df = df[["patientId", "bbox"]]
        df = df.groupby("patientId").agg(list)
        df = df.reset_index()
        df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)
        # create labels
        df["Target"] = df["bbox"].apply(lambda x: 0 if x == None else 1)
        
        assert split in ('train', 'val', 'test')
        with open(f'annotations/rsna/RSNAPneumonia_{split}.txt', 'r') as f:
            lines = f.readlines()
            image_ids = [l.split(' ')[0] for l in lines if l.strip() != '']
        self.df = df[df["patientId"].isin(image_ids)]

        if split == "train" and data_pct < 1.0:
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.transform = transform
        self.num_classes = 1

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(self.root, 'stage_2_train_images', row["patientId"]+'.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros([1024, 1024])
        bbox = row['bbox']
        if row['Target'] != 0:
            bbox = np.array(bbox)
            for i in range(len(bbox)):
                mask[bbox[i, 1]:bbox[i, 3],
                        bbox[i, 0]:bbox[i, 2]] += 1
        mask = (mask >= 1).astype("float32")
        # mask = resize_img(mask, self.imsize)
        augmented = self.transform(image=img, mask=mask)

        img = augmented["image"]
        mask = augmented["mask"].unsqueeze(0)
        return img, mask


if __name__ == '__main__':
    # proc rsna
    root_path = os.path.join(RSNA_DEFAULT_PATH, 'stage_2_train_images')
    for p in os.listdir(root_path):
        if '.dcm' not in p: continue
        print(p)
        img = dicom_to_pil_image(os.path.join(root_path, p))
        img.save(os.path.join(root_path, p.replace('.dcm', '.jpg')))
