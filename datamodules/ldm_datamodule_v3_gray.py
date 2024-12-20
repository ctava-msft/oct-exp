# -*- coding:utf-8 -*-
import json
import os

import cv2 as cv
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from natsort import natsorted
import numpy as np


class trainDatamodule(pl.LightningDataModule):
    def __init__(self, image_root, latent_root, train_name_json, test_name_json, batch_size=1, num_workers=8,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_root = image_root
        self.latent_root = latent_root
        with open(train_name_json, "r") as f:
            self.train_cube_names = json.load(f)
        with open(test_name_json, "r") as f:
            self.test_cube_names = json.load(f)

    def setup(self, stage=None):
        train_paths = []
        train_latent_paths = []
        for cube_name in self.train_cube_names:
            img_names = natsorted(os.listdir(os.path.join(self.img_root, cube_name)))
            for img_name in img_names:
                train_paths.append(os.path.join(self.img_root, cube_name, img_name))
                train_latent_paths.append(os.path.join(self.latent_root, cube_name, img_name[:-4]+'.npy'))
        test_paths = []
        test_latent_paths = []
        for cube_name in self.test_cube_names:
            img_names = natsorted(os.listdir(os.path.join(self.img_root, cube_name)))
            for img_name in img_names:
                test_paths.append(os.path.join(self.img_root, cube_name, img_name))
                test_latent_paths.append(os.path.join(self.latent_root, cube_name, img_name[:-4]+'.npy'))

        print(f'train len: {len(train_paths)}  test len: {len(test_paths)}')
        self.train_set = img_latent_Dataset(img_paths=train_paths, latent_paths=train_latent_paths)
        self.test_set = img_latent_Dataset(img_paths=test_paths, latent_paths=test_latent_paths)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class img_latent_Dataset(Dataset):
    def __init__(self, img_paths, latent_paths):
        self.img_paths = img_paths
        self.latent_paths = latent_paths

    def __getitem__(self, index):
        # print(self.img_paths[index], self.latent_paths[index])
        img = cv.imread(self.img_paths[index], cv.IMREAD_GRAYSCALE)
        splits = self.img_paths[index].split('/')
        img_id = int(splits[-1][:-4])
        if img_id == 1:
            pre_img = np.zeros_like(img)
        else:
            pre_name = str(img_id - 1) + '.png'
            pre_path = '/'.join(splits[:-1] + [pre_name])
            pre_img = cv.imread(pre_path, cv.IMREAD_GRAYSCALE)

        latent = np.load(self.latent_paths[index])
        latent = torch.from_numpy(latent).float()[0,:,:,:]
        # print(latent.shape)
        img = transforms.ToTensor()(img)
        img -= 0.5
        img /= 0.5
        pre_img = transforms.ToTensor()(pre_img)
        pre_img -= 0.5
        pre_img /= 0.5

        return {'image': img, 'latent':latent, 'pre_image': pre_img, 'img_path': self.img_paths[index],
                'latent_path': self.latent_paths[index]}

    def __len__(self):
        return len(self.img_paths)
