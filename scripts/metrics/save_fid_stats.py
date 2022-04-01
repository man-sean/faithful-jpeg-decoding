import argparse
import os
import pathlib

import torch
import numpy as np

from pytorch_fid.fid_score import IMAGE_EXTENSIONS, calculate_activation_statistics
from pytorch_fid.inception import InceptionV3

from scripts.data_preparation.create_lmdb import prepare_keys

IMAGE_EXTENSIONS.add('JPEG')


def my_compute_statistics_of_path(path, model, batch_size, dims, device,
                                  num_workers=1, suffix='JPEG'):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        # path = pathlib.Path(path)
        # files = sorted([file for ext in IMAGE_EXTENSIONS
        #                 for file in path.glob('*.{}'.format(ext))])
        files, _ = prepare_keys(path, suffix)
        print(f'{len(files)=}')
        print(files[:5])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--suffix', type=str, default='JPEG',
                        help='image suffix')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('-i', type=str, dest='in_path',
                        help='Path to the images')
    parser.add_argument('-o', type=str, dest='out_path',
                        help='Path to save statistics')
    args = parser.parse_args()

    if os.path.exists(args.out_path):
        raise argparse.ArgumentError(argument=None, message='Output path already exists')

    if not args.out_path.endswith('.npz'):
        raise argparse.ArgumentError(argument=None, message='Output path must end with .npz')

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx]).to(device)

    m, s = my_compute_statistics_of_path(
        path=args.in_path,
        model=model,
        batch_size=args.batch_size,
        dims=args.dims,
        device=device,
        num_workers=num_workers,
        suffix=args.suffix
    )

    with open(args.out_path, 'wb') as f:
        np.savez(f, mu=m, sigma=s)
