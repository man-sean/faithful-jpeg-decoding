import argparse
import cv2
import glob
import numpy as np
import os
import torch
import tqdm

from basicsr.utils.diffjpeg import DiffJPEG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qf', type=int, help='JPEG QF')
    parser.add_argument('--input', type=str, help='input test image folder')
    parser.add_argument('--output', type=str, help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = DiffJPEG(differentiable=True).to(device)

    os.makedirs(args.output, exist_ok=False)
    for idx, path in tqdm.tqdm(enumerate(sorted(glob.glob(os.path.join(args.input, '*'))))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        # print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                output = model(img, quality=args.qf)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_QF{args.qf}.png'), output)


if __name__ == '__main__':
    main()
