# -*- coding:utf-8 -*-
import json
import glob
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
    def __init__(self,  latent_root, train_name_json, test_name_json, batch_size=1, num_workers=8,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.latent_root = latent_root
        with open(train_name_json, "r") as f:
            self.train_cube_names = json.load(f)
        with open(test_name_json, "r") as f:
            self.test_cube_names = json.load(f)

    def setup(self, stage=None):
        print(f'latent_root: {self.latent_root}  train_cube_names: {self.train_cube_names}')
        train_latent_paths = []
        for cube_name in self.train_cube_names:
            npy_files = glob.glob(os.path.join(self.latent_root, cube_name, '*.npy'))
            for npy_file in npy_files:
                train_latent_paths.append(npy_file)

        test_latent_paths = []
        for cube_name in self.test_cube_names:
            npy_files = glob.glob(os.path.join(self.latent_root, cube_name, '*.npy'))
            for npy_file in npy_files:
                test_latent_paths.append(npy_file)

        print(f'train len: {len(train_latent_paths)}  test len: {len(test_latent_paths)}')
        self.train_set = latent_Dataset(latent_paths=train_latent_paths)
        self.test_set = latent_Dataset(latent_paths=test_latent_paths)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class testDatamodule(pl.LightningDataModule):
    def __init__(self,  latent_root, batch_size=1, num_workers=8,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.latent_root = latent_root

    def setup(self, stage=None):
        self.test_cube_names = os.listdir(self.latent_root)
        test_latent_paths = []
        for cube_name in self.test_cube_names:
            test_latent_paths.append(os.path.join(self.latent_root, cube_name))

        print(f'  test len: {len(test_latent_paths)}')
        self.test_set = latent_Dataset(latent_paths=test_latent_paths)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class latent_Dataset(Dataset):
    def __init__(self, latent_paths):
        self.latent_paths = latent_paths

    def __getitem__(self, index):
        print(self.latent_paths[index])

        latent = np.load(self.latent_paths[index])
        latent = torch.from_numpy(latent).float()[0,:,:]

        return {'latent':latent,
                'latent_path': self.latent_paths[index]}

    def __len__(self):
        return len(self.latent_paths)
