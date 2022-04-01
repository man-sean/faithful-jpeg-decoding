import argparse
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import os.path as osp
import glob
import os

from tqdm import tqdm


def encode_worker(path, lr_folder):
    base = osp.splitext(osp.basename(path))[0]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    cv2.imwrite(osp.join(lr_folder, f'{base}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, args.qf])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--suffix', type=str, default='JPEG',
                        help='image suffix')
    parser.add_argument('-i', type=str, dest='in_path', required=True,
                        help='Path to the images')
    parser.add_argument('-o', type=str, dest='out_path', required=True,
                        help='Path to save compressed images')
    parser.add_argument('-q', type=int, dest='qf', required=True,
                        help='QF in [0,100]')
    args = parser.parse_args()

    if os.path.exists(args.out_path):
        raise argparse.ArgumentError(argument=None, message='Output path already exists')

    hr_folder = osp.join(args.in_path, f'*.{args.suffix}')
    lr_folder = args.out_path
    print(f'Uncompressed folder: {hr_folder}')
    print(f'Compressed folder: {lr_folder}')
    if not osp.exists(lr_folder):
        print(f'create compressed folder')
        os.makedirs(lr_folder)

    idx = 0
    length = len(glob.glob(hr_folder, recursive=True))
    pbar = tqdm(total=length, unit='image', desc='Compress')

    def callback(arg):
        """get the image data and update pbar."""
        pbar.update(1)

    pool = Pool(10)
    for path in tqdm(glob.glob(hr_folder, recursive=True), total=length):
        pool.apply_async(encode_worker, args=(path, lr_folder), callback=callback)
    pool.close()
    pool.join()
    pbar.close()


    # print(f'{length=}')
    # for path in tqdm(glob.glob(hr_folder, recursive=True), total=length):
    #     idx += 1
    #     base = osp.splitext(osp.basename(path))[0]
    #     img = cv2.imread(path, cv2.IMREAD_COLOR)
    #     cv2.imwrite(osp.join(lr_folder, f'{base}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, args.qf])