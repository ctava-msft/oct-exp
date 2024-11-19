import argparse
import cv2
import glob
import numpy as np

def main(path):
    # Load slices
    slice_files = sorted(glob.glob(f'{path}/*.bmp'))  # Replace 'path_to_slices' with the actual path
    slices = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in slice_files]
    # Stack slices to form a 3D volume
    volume = np.stack(slices, axis=0)  # Volume shape: (num_slices, height, width)
    # Create a 2D projection (Maximum Intensity Projection)
    # You can change this to np.mean(volume, axis=0) for average projection
    projection = np.max(volume, axis=0)
    # Save the projection as a .png file
    output_path = 'volume_projection.png'
    cv2.imwrite(output_path, projection)
    print(f"Volume projection saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create volume projection from image slices.')
    parser.add_argument('path', help='Path to the directory containing the image slices.')
    args = parser.parse_args()
    main(args.path)