import argparse
import os
import glob
import cv2
import numpy as np
import torch

def read_cube_to_np(path, num_channels=8, cvflag=cv2.IMREAD_GRAYSCALE):
    """
    Reads a specified number of .bmp images from a directory, converts them to grayscale,
    and stacks them into a 3D numpy array.

    Args:
        path (str): Directory containing the .bmp image slices.
        num_channels (int): Number of image slices to read and stack.
        cvflag: OpenCV flag for image reading (default is grayscale).

    Returns:
        np.ndarray: Stacked 3D volume with shape (num_channels, height, width).
    """
    # Load slice file paths, sorted to maintain order
    slice_files = sorted(glob.glob(os.path.join(path, '*.bmp')))[:num_channels]
    if len(slice_files) < num_channels:
        raise ValueError(f"Expected at least {num_channels} .bmp files in {path}, but found {len(slice_files)}.")

    slices = []
    for img_path in slice_files:
        img = cv2.imread(img_path, cvflag)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # Ensure the image is single-channel
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            pass  # Already single-channel
        else:
            raise ValueError(f"Unexpected image shape: {img.shape} for image {img_path}")

        slices.append(img)  # Each img has shape [640, 400]

    # Verify all slices have the same dimensions
    heights = [s.shape[0] for s in slices]
    widths = [s.shape[1] for s in slices]
    if len(set(heights)) != 1 or len(set(widths)) != 1:
        raise ValueError("Not all image slices have the same dimensions.")

    # Stack slices to form a 3D volume [num_channels, height, width]
    volume = np.stack(slices, axis=0)  # Shape: [8, 640, 400]
    return volume

def make_volume(path, num_channels=8):
    """
    Creates a PyTorch tensor representing the volume from image slices.

    Args:
        path (str): Directory containing the .bmp image slices.
        num_channels (int): Number of image slices to read and stack.

    Returns:
        torch.Tensor: Tensor with shape [1, num_channels, height, width].
    """
    # Read and stack image slices into a numpy array
    volume_np = read_cube_to_np(path, num_channels=num_channels, cvflag=cv2.IMREAD_GRAYSCALE)
    print(f"Numpy volume shape: {volume_np.shape}")  # Should be [8, 640, 400]

    # Convert numpy array to torch tensor
    volume_tensor = torch.from_numpy(volume_np).float()  # Shape: [8, 640, 400]
    print(f"Tensor shape after conversion: {volume_tensor.shape}")  # Should be [8, 640, 400]

    # Add a batch dimension to make it [1, 8, 640, 400]
    volume_tensor = volume_tensor.unsqueeze(0)  # Shape: [1, 8, 640, 400]
    print(f"Tensor shape after unsqueeze: {volume_tensor.shape}")  # Should be [1, 8, 640, 400]

    return volume_tensor

def main(path, num_channels=8):
    volume_tensor = make_volume(path, num_channels=num_channels)
    output_path = os.path.join(path, 'volume_projection.npy')
    np.save(output_path, volume_tensor.numpy())
    print(f"Volume projection saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create volume projection from image slices.')
    parser.add_argument('path', help='Path to the directory containing the image slices.')
    parser.add_argument('--num_channels', type=int, default=8, help='Number of image slices to read and stack.')
    args = parser.parse_args()
    main(args.path, args.num_channels)