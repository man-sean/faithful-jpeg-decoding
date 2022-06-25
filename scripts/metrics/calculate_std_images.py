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
    img_list_gt = sorted(list(scandir(args.restored[0], recursive=True, full_path=True)))
    img_list_restored = [sorted(list(scandir(restored_dir, recursive=True, full_path=True))) for restored_dir in args.restored]

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))

        std_fake = []
        for restored_dir in img_list_restored:
            bgr_img = cv2.imread(restored_dir[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            rgb_img = img2tensor(bgr_img, bgr2rgb=True, float32=True)
            std_fake.append(rgb_img)

        # generate std
        std_fake = torch.stack(std_fake, dim=0)
        std_fake = rgb_to_ycbcr(std_fake)[..., 0:1, :, :]
        std = std_fake.std(0)
        caption = [f'STD stats: {std[idx].mean():.4} ±{std[idx].std():.4} [{std[idx].min():.3}, {std[idx].max():.3}]'
                   for idx in range(std.shape[0])]
        std = std.pow(1 / 4).expand(3, -1, -1).clone()
        std = 1 - np.einsum('...chw->...hwc', std.detach().clamp_(0, 1).cpu().numpy())
        std = (std * 255.0).round()

        imwrite(std, str(args.output / f'{basename}.png'))
        print(f'{caption}')
    print(args.restored)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restored', type=str, required=True, nargs='+', help='Path to restored images')
    parser.add_argument('--output', type=Path, required=True, help='Path to save std images')
    args = parser.parse_args()
    main(args)
