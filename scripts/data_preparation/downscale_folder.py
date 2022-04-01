import cv2
import numpy as np
import torch
import os.path as osp
import glob
import os

from tqdm import tqdm

from basicsr.utils.matlab_functions import imresize
from torch.nn.functional import interpolate
import torch.nn.functional as F
from basicsr.utils.img_util import img2tensor, tensor2img


hr_folder = '/home/sean.man/datasets/ffhq_128/test/dummy/*.png'
lr_folder = '/home/sean.man/datasets/ffhq_64/test/dummy'
print(f'HR folder: {hr_folder}')
print(f'LR folder: {lr_folder}')
if not osp.exists(lr_folder):
    print(f'create LR folder')
    os.makedirs(lr_folder)
scale = 2

idx = 0
length = len(glob.glob(hr_folder, recursive=True))
for path in tqdm(glob.glob(hr_folder, recursive=True), total=length):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = img2tensor(img).unsqueeze(0)

    # If you want to use matlab downsampling
    # lr_img = imresize(img, 1.0 / scale)
    # lr_img = imresize(img, 1.0 / scale)
    lr_img = interpolate(img, scale_factor=1.0 / scale, mode='nearest')
    lr_img = tensor2img(lr_img.squeeze(0))
    # lr_img = img

    # # If you want to use pytorch downsampling
    # img = img2tensor()

    # print(base, path)
    # cv2.imwrite(f'{lr_folder}/{base}x{scale}.png', lr_img)
    cv2.imwrite(f'{lr_folder}/{base}.png', lr_img)
