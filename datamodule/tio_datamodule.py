# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import torchio as tio
from torchvision.utils import save_image
import numpy as np
import torch
from natsort import natsorted
import json
import SimpleITK as sitk
import nibabel as nib

def image_reader(path):
    data = np.load(path, allow_pickle=True)
    data = torch.from_numpy(data).float()
    data /= 255.
    data = data.unsqueeze(0)
    affine = np.eye(4)
    return data, affine

def image_reader_new(path):
    try:
        data = np.load(path, allow_pickle=True)
        data = data.astype(np.float32)
        sitk_image = sitk.GetImageFromArray(data)
        return sitk_image
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def label_reader(path):
    data = np.load(path, allow_pickle=True)
    data = torch.from_numpy(data).long()
    data //= 255
    data = data.unsqueeze(0)
    affine = np.eye(4)
    return data, affine


def is_valid_numpy_file(file_path):
    try:
        array = np.load(file_path, allow_pickle=True)

        # # Ensure the array has a numeric data type
        # if array.dtype == object:
        #     array = array.astype(np.float32)

        # # Check the number of dimensions of the array
        # if array.ndim == 1:
        #     # Reshape the array to have at least 2 dimensions
        #     array = array.reshape((1, -1))
        # elif array.ndim == 2:
        #     # If the array is 2D, ensure it has the correct shape for an image
        #     array = array.reshape((array.shape[0], array.shape[1], 1))
        # # Convert the numpy array to a NiBabel image
        # nib_image = nib.Nifti1Image(array, affine=np.eye(4))

        # # Convert the NiBabel image to a SimpleITK image
        # sitk_image = sitk.GetImageFromArray(nib_image.get_fdata())
        return True
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False
    
class TioDatamodule(pl.LightningDataModule):
    def __init__(self, image_npy_root, train_name_json, test_name_json, image_size, batch_size, num_workers, patch_per_size,
                 queue_length, samples_per_volume, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_root = image_npy_root
        self.train_subjects = []
        self.test_subjects = []
        self.image_size = image_size
        self.patch_size = patch_per_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume

        with open(train_name_json, "r") as f:
            self.train_image_names = json.load(f)
        with open(test_name_json, "r") as f:
            self.test_image_names = json.load(f)

    def prepare_data(self):
        # Prepare training subjects
        for name in self.train_image_names:
            folder_path = os.path.join(self.image_root, name)
            
            # Check if the path is a directory
            if os.path.isdir(folder_path):
                # Loop through all .npy files in the folder
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.npy'):
                        file_path = os.path.join(folder_path, file_name)
                        # Load the .npy file using torchio
                        try:
                            if is_valid_numpy_file(file_path):
                                image = tio.ScalarImage(file_path)
                                subject = tio.Subject(image=image, name=name)
                                self.train_subjects.append(subject)
                                print(f"Subject Train Append: {file_path}")
                            else:
                                print(f"Invalid file: {file_path}")
                        except FileNotFoundError:
                            print(f"File not found: {file_path}")
                        except ValueError as e:
                            print(f"Error loading file {file_path}: {e}")

            # Prepare test subjects
            for name in self.test_image_names:
                folder_path = os.path.join(self.image_root, name)
                
                # Check if the path is a directory
                if os.path.isdir(folder_path):
                    # Loop through all .npy files in the folder
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith('.npy'):
                            file_path = os.path.join(folder_path, file_name)
                            # Load the .npy file using torchio
                            try:
                                if is_valid_numpy_file(file_path):
                                    image = tio.ScalarImage(file_path)
                                    subject = tio.Subject(image=image, name=name)
                                    self.test_subjects.append(subject)
                                    print(f"Subject Test Append: {file_path}")
                                else:
                                    print(f"Invalid file: {file_path}")
                            except FileNotFoundError:
                                print(f"File not found: {file_path}")
                            except ValueError as e:
                                print(f"Error loading file {file_path}: {e}")

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.Resize(self.image_size),
            tio.RescaleIntensity((-1, 1), in_min_max=(0, 1)),
        ])
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomFlip(axes=(0, 2)),
        ])
        return augment

    def setup(self, stage=None):
        preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        train_set = tio.SubjectsDataset(self.train_subjects, transform=tio.Compose([preprocess, augment]))
        self.patch_train_set = tio.Queue(
            train_set,
            self.queue_length,
            self.samples_per_volume,
            tio.data.UniformSampler(self.patch_size),
            num_workers=self.num_workers,
            shuffle_patches=True,
            shuffle_subjects=True
        )
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=preprocess)

    def train_dataloader(self):
        return DataLoader(self.patch_train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
