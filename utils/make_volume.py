import argparse
import cv2
import glob
import numpy as np
import torch

def read_cube_to_np(path, num_channels=8, cvflag=cv2.IMREAD_GRAYSCALE):
    # Load slices
    slice_files = sorted(glob.glob(f'{path}/*.bmp'))[:num_channels]  # Limit to num_channels files
    slices = [cv2.imread(img, cvflag) for img in slice_files]
    
    # Ensure each slice is a single-channel image
    slices = [slice[np.newaxis, :, :] for slice in slices]  # Add channel dimension
    
    # Stack slices to form a 3D volume
    volume = np.stack(slices, axis=0)  # Volume shape: (num_channels, height, width)
    
    return volume

def make_volume(path, num_channels=8):
    # Read images and convert to numpy array
    volume_np = read_cube_to_np(path, num_channels=num_channels, cvflag=cv2.IMREAD_GRAYSCALE)
    
    # Convert numpy array to torch tensor
    volume_tensor = torch.from_numpy(volume_np).float()
    
    # Ensure the tensor has the correct shape [1, num_channels, height, width]
    volume_tensor = volume_tensor.unsqueeze(1)  # Add batch dimension, shape becomes [num_channels, 1, height, width]
    volume_tensor = volume_tensor.permute(1, 0, 2, 3)  # Change shape to [1, num_channels, height, width]
    
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