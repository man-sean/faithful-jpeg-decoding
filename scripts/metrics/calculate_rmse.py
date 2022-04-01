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
    img_list_restored = sorted(list(scandir(args.restored, recursive=True, full_path=True)))

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # if args.suffix == '':
        #     img_path_restored = img_list_restored[i]
        # else:
        #     img_path_restored = osp.join(args.restored, basename + args.suffix + ext)
        img_path_restored = osp.join(args.restored, basename + args.suffix + ext)
        if not osp.exists(img_path_restored):
            print(f'DOES NOT EXISTS: {img_path_restored}, skipping')
            continue
        # else:
        #     print(f'FOUND: {img_path_restored} <-> {img_path}')
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        # if args.correct_mean_var:
        #     mean_l = []
        #     std_l = []
        #     for j in range(3):
        #         mean_l.append(np.mean(img_gt[:, :, j]))
        #         std_l.append(np.std(img_gt[:, :, j]))
        #     for j in range(3):
        #         # correct twice
        #         mean = np.mean(img_restored[:, :, j])
        #         img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
        #         std = np.std(img_restored[:, :, j])
        #         img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]
        #
        #         mean = np.mean(img_restored[:, :, j])
        #         img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
        #         std = np.std(img_restored[:, :, j])
        #         img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate RMSE
        psnr = calculate_rmse(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        print(f'{i+1:3d}: {basename:25}. \tRMSE: {psnr:.6f}')
        rmse_all.append(psnr)
    print(args.gt)
    print(args.restored)
    print(f'Average: RMSE: {sum(rmse_all) / len(rmse_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    # parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    main(args)
