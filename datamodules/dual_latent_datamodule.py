# -*- coding:utf-8 -*-
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
from natsort import natsorted
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class trainDatamodule(pl.LightningDataModule):
    def __init__(self, latent_3D_root, latent_2D_root, train_name_json, test_name_json, batch_size=1, num_workers=8,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.latent_3D_root = latent_3D_root
        self.latent_2D_root = latent_2D_root
        with open(train_name_json, "r") as f:
            self.train_cube_names = json.load(f)
        with open(test_name_json, "r") as f:
            self.test_cube_names = json.load(f)

    def setup(self, stage=None):
        train_latent_3D_paths = []
        train_latent_2D_paths = []
        for cube_name in self.train_cube_names:
            train_latent_3D_paths.append(os.path.join(self.latent_3D_root, cube_name + '.npy'))
            train_latent_2D_paths.append(os.path.join(self.latent_2D_root, cube_name + '.npy'))

        print(f'train len: {len(train_latent_3D_paths)} {len(train_latent_2D_paths)}')
        self.train_set = dual_latent_Dataset(latent_3D_paths=train_latent_3D_paths,
                                             latent_2D_paths=train_latent_2D_paths)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)


class dual_latent_Dataset(Dataset):
    def __init__(self, latent_3D_paths, latent_2D_paths):
        self.latent_3D_paths = latent_3D_paths
        self.latent_2D_paths = latent_2D_paths

    def __getitem__(self, index):
        # print(self.img_paths[index], self.latent_paths[index])

        latent_3D = np.load(self.latent_3D_paths[index])
        latent_3D = torch.from_numpy(latent_3D).float()[0, :, :, :]

        latent_2D = np.load(self.latent_2D_paths[index])
        latent_2D = torch.from_numpy(latent_2D).float()[0, :, :, :]
        return {'latent_3D': latent_3D, 'latent_2D': latent_2D,
                'latent_2D_path': self.latent_2D_paths[index],
                'latent_3D_path': self.latent_3D_paths[index]}

    def __len__(self):
        return len(self.latent_3D_paths)


class test_single_latent_Datamodule(pl.LightningDataModule):
    def __init__(self, latent_root, batch_size=1, num_workers=4,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.latent_root = latent_root

    def setup(self, stage=None):
        latent_paths = []
        cube_names = natsorted(os.listdir(self.latent_root))
        for cube_name in cube_names:
            latent_paths.append(os.path.join(self.latent_root, cube_name))

        print(f'test len: {len(latent_paths)} ')
        self.train_set = single_latent_Dataset(latent_3D_paths=latent_paths)

    def test_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class single_latent_Dataset(Dataset):
    def __init__(self, latent_3D_paths):
        self.latent_3D_paths = latent_3D_paths

    def __getitem__(self, index):
        # print(self.img_paths[index], self.latent_paths[index])

        latent_3D = np.load(self.latent_3D_paths[index])
        latent_3D = torch.from_numpy(latent_3D).float()[0, :, :, :]
        return {'latent_3D': latent_3D,
                'latent_3D_path': self.latent_3D_paths[index]}

    def __len__(self):
        return len(self.latent_3D_paths)
