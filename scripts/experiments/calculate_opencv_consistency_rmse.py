import argparse
import cv2
import numpy as np
from os import path as osp

from basicsr.metrics.psnr_ssim import calculate_rmse
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr


def main(args):
    """Calculate RMSE for images.
    """
    rmse_all = []
    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))

    qf = 100
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qf, int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR)]

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        _, encimg = cv2.imencode('.jpg', img_gt, encode_param)
        img_restored = cv2.imdecode(encimg, 1).astype(np.float32) / 255.
        img_gt = img_gt.astype(np.float32) / 255.

        # calculate RMSE
        rmse = calculate_rmse(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        print(f'{i+1:3d}: {basename:25}. \tRMSE: {rmse:.6f}')
        rmse_all.append(rmse)
    print(args.gt)
    print(f'Average: RMSE: {sum(rmse_all) / len(rmse_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, help='Path to gt (Ground-Truth)')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    args = parser.parse_args()
    main(args)
