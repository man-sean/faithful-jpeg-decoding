import os
import argparse
from pathlib import Path

import numpy as np
import torch

from pytorch_fid.fid_score import calculate_fid_given_paths, IMAGE_EXTENSIONS
from pytorch_fid.inception import InceptionV3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--real', type=Path, required=True, help='Path to real samples folder or .npz file.')
    parser.add_argument('-f', '--fake', type=Path, required=True, nargs='+', help='Path to fake samples folders.')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    fids = []

    for idx, fake_path in enumerate(args.fake):
        print(f'[>] Calculating {idx + 1}/{len(args.fake)} realizations')
        fids.append(calculate_fid_given_paths([str(args.real), str(fake_path)],
                                              args.batch_size,
                                              device,
                                              args.dims,
                                              num_workers))
        print(f'[>] This round FID: {fids[-1]}')

    print(f'[>] Calculating statistics...')
    fids = np.array(fids)
    fid_mean = fids.mean()
    fid_std = fids.std()
    print(f"FID, repeated {len(args.fake)} times: {fid_mean} ±{fid_std}")


if __name__ == '__main__':
    main()
