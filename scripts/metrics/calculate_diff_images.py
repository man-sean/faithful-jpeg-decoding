import argparse
from pathlib import Path

import cv2
import numpy as np
from os import path as osp

import torch

from basicsr import img2tensor, tensor2img, imwrite
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.collage_util import rgb_to_ycbcr
from basicsr.utils.matlab_functions import bgr2ycbcr


def main(args):
    """Calculate PSNR and SSIM for images.
    """
    img_list_gt = sorted(list(scandir(args.compressed, recursive=True, full_path=True)))
    img_list_restored = sorted(list(scandir(args.recompressed, recursive=True, full_path=True)))

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))

        gt_img = img2tensor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), bgr2rgb=True, float32=True) / 255.
        img_path_restored = osp.join(args.recompressed, basename + ext)
        if not osp.exists(img_path_restored):
            print(f'DOES NOT EXISTS: {img_path_restored}, skipping')
            continue
        # else:
        #     print(f'FOUND: {img_path_restored} <-> {img_path}')
        img_restored = img2tensor(cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED), bgr2rgb=True, float32=True) / 255.

        diff = (gt_img - img_restored).abs() ** (1 / 4)
        diff = tensor2img(diff)

        imwrite(diff, str(args.output / f'{basename}.png'))
        print(f'{i}: mean={diff.mean()}')
    print(args.recompressed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compressed', type=str, required=True, help='Path to compressed images')
    parser.add_argument('--recompressed', type=str, required=True, help='Path to recompressed images')
    parser.add_argument('--output', type=Path, required=True, help='Path to save diff images')
    args = parser.parse_args()
    main(args)
