import argparse
import cv2
import glob
import os
from natsort import natsorted
import numpy as np
import torch

def read_cube_to_np(img_dir, stack_axis=0, cvflag=cv2.IMREAD_GRAYSCALE):
    assert os.path.exists(img_dir), f"got {img_dir}"
    print(img_dir)
    imgs = []
    names = natsorted(os.listdir(img_dir))
    for name in names:
        img = cv2.imread(os.path.join(img_dir, name), cvflag)
        imgs.append(img)
    imgs = np.stack(imgs, axis=stack_axis)
    return imgs

def orig(path):
    # Load slices
    slice_files = sorted(glob.glob(f'{path}/*.bmp'))  # Replace 'path_to_slices' with the actual path
    slices = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in slice_files]
    # Stack slices to form a 3D volume
    volume = np.stack(slices, axis=0)  # Volume shape: (num_slices, height, width)
    # Create a 2D projection (Maximum Intensity Projection)
    # You can change this to np.mean(volume, axis=0) for average projection
    projection = np.max(volume, axis=0)
    # Save the projection as a .npy file
    output_path = f'{path}/volume_projection.npy'
    np.save(output_path, projection)
    print(f"Volume projection saved as {output_path}")

def mainOld(path):
    volume = read_cube_to_np(path, stack_axis=0, cvflag=cv2.IMREAD_GRAYSCALE)
    output_path = f'{path}/volume_projection.npy'
    np.save(output_path, volume)
    print(f"Volume projection saved as {output_path}")


def make_volume(path, num_channels=8):
    # Read images and convert to numpy array
    volume_np = read_cube_to_np(path, num_channels=num_channels, cvflag=cv2.IMREAD_GRAYSCALE)
    
    # Convert numpy array to torch tensor
    volume_tensor = torch.from_numpy(volume_np).float()
    
    # Ensure the tensor has the correct shape [batch_size, channels, height, width]
    # Here we assume batch_size=1 and stack along the channel dimension
    volume_tensor = volume_tensor.permute(1, num_channels, 1280, 400)  # Change shape to [1, num_channels, height, width]
    
    return volume_tensor

def main(path):
    volume_tensor = make_volume(path)
    output_path = f'{path}/volume_projection.npy'
    np.save(output_path, volume_tensor.numpy())
    print(f"Volume projection saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create volume projection from image slices.')
    parser.add_argument('path', help='Path to the directory containing the image slices.')
    args = parser.parse_args()
    main(args.path)