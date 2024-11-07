import os
import cv2 as cv
from natsort import natsorted
import numpy as np

def read_img_to_np(img_dir, name, cvflag =  cv.IMREAD_GRAYSCALE):
    assert os.path.exists(img_dir), f"got {img_dir}"
    print(img_dir)
    imgs = []
    names = natsorted(os.listdir(img_dir))
    print(names)
    for name in names:
        img = cv.imread(os.path.join(img_dir, name),cvflag)
        imgs.append(img)
    imgs = np.stack(imgs, axis=0)
    return imgs

def save(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    names = natsorted(os.listdir(src_dir))
    for i, name in enumerate(names):
        print(i, name)
        oct = read_img_to_np(src_dir, name, cv.IMREAD_GRAYSCALE)
        print(oct.shape, np.max(oct))
        np.save(os.path.join(dst_dir, name+'.npy'), oct)

if __name__ == '__main__':
    src_dir = '.\images'
    dst_dir = '.\images_converted'
    save(src_dir, dst_dir)
