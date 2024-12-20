# -*- coding:utf-8 -*-
import json
import os

import albumentations as A
import cv2 as cv
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from natsort import natsorted


class trainDatamodule(pl.LightningDataModule):
    def __init__(self, data_root, train_name_json, test_name_json, image_size, crop_size, batch_size=1, num_workers=8,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        with open(train_name_json, "r") as f:
            self.train_cube_names = json.load(f)
        with open(test_name_json, "r") as f:
            self.test_cube_names = json.load(f)
        self.train_transform = A.Compose([
            A.Resize(*image_size),
            A.RandomCrop(*crop_size),
            A.Normalize((0.5,), (0.5,))
        ])
        self.test_transform = A.Compose([
            A.Resize(*image_size),
            A.Normalize((0.5,), (0.5,))
        ])

    def setup(self, stage=None):
        train_paths = []
        for cube_name in self.train_cube_names:
            img_names = natsorted([
                img_name for img_name in os.listdir(os.path.join(self.data_root, cube_name))
                if img_name.lower().endswith('.bmp')
            ])
            train_paths += [os.path.join(self.data_root, cube_name, img_name) for img_name in img_names]
        test_paths = []
        for cube_name in self.test_cube_names:
            img_names = natsorted([
                img_name for img_name in os.listdir(os.path.join(self.data_root, cube_name))
                if img_name.lower().endswith('.bmp')
            ])
            test_paths += [os.path.join(self.data_root, cube_name, img_name) for img_name in img_names]
        print(f'train len: {len(train_paths)}  test len: {len(test_paths)}')
        self.train_set = unlabeled_Dataset(paths=train_paths, transform=self.train_transform)
        self.test_set = unlabeled_Dataset(paths=test_paths, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class unlabeled_Dataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        img = cv.imread(self.paths[index], cv.IMREAD_GRAYSCALE)

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']

        img = transforms.ToTensor()(img)

        return {'image': img, 'path': self.paths[index]}

    def __len__(self):
        return len(self.paths)
