import argparse
import cv2
import numpy as np
from os import path as osp
import tempfile
from PIL import Image

from basicsr.metrics.psnr_ssim import calculate_rmse, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr


def main(args):
    """Calculate RMSE for images.
    """
    rmse_all = []
    ssim_all = []
    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))

    qtables = [
        [1] * 64,
        [1] * 64,
    ]

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = Image.open(img_path)

        with tempfile.NamedTemporaryFile() as tmp:
            img_gt.save(tmp.name, format="JPEG", subsampling=0, qtables=qtables)
            img_restored = np.array(Image.open(tmp.name)).astype(np.float32) / 255.
        img_gt = np.array(img_gt).astype(np.float32) / 255.

        # calculate RMSE
        rmse = calculate_rmse(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        print(f'{i+1:3d}: {basename:25}. \tRMSE: {rmse:.6f}, \tSSIM: {ssim:.6f}')
        rmse_all.append(rmse)
        ssim_all.append(ssim)
    print(args.gt)
    print(f'Average: RMSE: {sum(rmse_all) / len(rmse_all):.6f}, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, help='Path to gt (Ground-Truth)')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    args = parser.parse_args()
    main(args)
