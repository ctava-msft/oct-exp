import argparse
import cv2 as cv
import numpy as np
from natsort import natsorted
import os

def read_img_to_np(img_dir, cvflag=cv.IMREAD_GRAYSCALE):
    assert os.path.exists(img_dir), f"Directory does not exist: {img_dir}"
    print(f"Reading images from directory: {img_dir}")
    imgs = []
    names = natsorted([name for name in os.listdir(img_dir) if name.endswith('.bmp')])
    #print(f"Found images: {names}")
    for name in names:
        img_path = os.path.join(img_dir, name)
        img = cv.imread(img_path, cvflag)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        imgs.append(img)
    imgs = np.stack(imgs, axis=0)
    return imgs

def save(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    names = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            names.append(os.path.join(root, file))
    names = natsorted(names)
    for i, name in enumerate(names):
        print(f"Processing file {i}: {name}")
        oct_data = read_img_to_np(os.path.dirname(name), cv.IMREAD_GRAYSCALE)
        base_name = os.path.splitext(os.path.basename(name))[0]  # Remove the extension
        np.save(os.path.join(dst_dir, base_name + '.npy'), oct_data)
        print(f"Saved {base_name}.npy to {dst_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert images to numpy files.")
    parser.add_argument('src_dir', type=str, help="Source directory containing images.")
    parser.add_argument('dst_dir', type=str, help="Destination directory to save numpy files.")
    args = parser.parse_args()
    
    save(args.src_dir, args.dst_dir)

if __name__ == "__main__":
    main()