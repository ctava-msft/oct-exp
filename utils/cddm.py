import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_clip_embeddings(image_paths):
    images = [preprocess(Image.open(image_path)).unsqueeze(0).to(device) for image_path in image_paths]
    images = torch.cat(images, dim=0)
    with torch.no_grad():
        embeddings = model.encode_image(images)
    return embeddings.cpu().numpy()

def compute_mmd(x, y):
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)
    rx = np.diag(xx)
    ry = np.diag(yy)
    k = np.exp(-0.5 * (rx[:, None] + rx[None, :] - 2 * xx))
    l = np.exp(-0.5 * (ry[:, None] + ry[None, :] - 2 * yy))
    m = np.exp(-0.5 * (rx[:, None] + ry[None, :] - 2 * xy))
    return np.mean(k) + np.mean(l) - 2 * np.mean(m)

def calculate_cmmd(image_paths1, image_paths2):
    embeddings1 = get_clip_embeddings(image_paths1)
    embeddings2 = get_clip_embeddings(image_paths2)
    cmmd = compute_mmd(embeddings1, embeddings2)
    return cmmd

image_paths1 = ["./images/0-before.png"]
image_paths2 = ["./images/0-after.png"]
cmmd_value = calculate_cmmd(image_paths1, image_paths2)
print(f"CMMD value: {cmmd_value}")