import argparse
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import os.path as osp
import glob
import os

from torchvision.io import read_image, write_png
from torchvision.transforms.functional import center_crop

from tqdm import tqdm


def crop_worker(path, crop_size, lr_folder):
    base = osp.splitext(osp.basename(path))[0]
    img = read_image(path)
    img = center_crop(img, crop_size)
    write_png(img, osp.join(lr_folder, f'{base}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--suffix', type=str, default='png',
                        help='image suffix')
    parser.add_argument('-i', type=str, dest='in_path', required=True,
                        help='Path to the images')
    parser.add_argument('-o', type=str, dest='out_path', required=True,
                        help='Path to save cropped images')
    parser.add_argument('-c', type=int, dest='crop_size', required=True,
                        help='center crop size')
    args = parser.parse_args()

    if os.path.exists(args.out_path):
        raise argparse.ArgumentError(argument=None, message='Output path already exists')

    hr_folder = osp.join(args.in_path, f'*.{args.suffix}')
    lr_folder = args.out_path
    print(f'Uncropped folder: {hr_folder}')
    print(f'Cropped folder: {lr_folder}')
    if not osp.exists(lr_folder):
        print(f'create cropped folder')
        os.makedirs(lr_folder)

    idx = 0
    length = len(glob.glob(hr_folder, recursive=True))
    pbar = tqdm(total=length, unit='image', desc='Cropping')

    def callback(arg):
        """get the image data and update pbar."""
        pbar.update(1)

    pool = Pool(10)
    for path in tqdm(glob.glob(hr_folder, recursive=True), total=length):
        pool.apply_async(crop_worker, args=(path, args.crop_size, lr_folder), callback=callback)
    pool.close()
    pool.join()
    pbar.close()