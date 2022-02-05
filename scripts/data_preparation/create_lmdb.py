import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool
from os import path as osp
from pathlib import Path

import cv2
import lmdb
import tqdm
from torchvision.datasets import ImageFolder

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import read_img_worker


def create_lmdb(folder_path, lmdb_path, suffix):
    """Create lmdb files for ImageNet dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Remember to modify opt configurations according to your settings.
    """
    print('Creating LMDB keys', flush=True)
    img_path_list, keys = prepare_keys(folder_path, suffix)
    print('Creating LMDB files', flush=True)
    start_time = time.time()
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)
    end_time = time.time()
    print(f'total time: {end_time - start_time} seconds', flush=True)


def prepare_keys(folder_path, suffix):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...', flush=True)

    with os.scandir(folder_path) as scanit:
        subdirs = [x.path for x in scanit if x.is_dir()]
    if len(subdirs) == 0:
        subdirs = [folder_path]
    print(f'Found {len(subdirs)} subdirs', flush=True)

    img_path_list = []
    for subdir in tqdm.tqdm(subdirs, file=sys.stdout):
        img_path_list += list(glob.glob(os.path.join(subdir, f'*.{suffix}')))
        # img_path_list += list(scandir(subdir, suffix='png', recursive=False))

    print(f'Sorting paths', flush=True)
    img_path_list = sorted(img_path_list)

    print('Extracting keys', flush=True)
    keys = [img_path.split(f'.{suffix}')[0] for img_path in img_path_list]

    return img_path_list, keys


def make_lmdb_from_imgs(data_path,
                        lmdb_path,
                        img_path_list,
                        keys,
                        batch=5000,
                        mp_batch=100000,
                        compress_level=1,
                        multiprocessing_read=False,
                        n_thread=40,
                        map_size=None):
    """Make lmdb from images.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.

    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1

    We use the image name without extension as the lmdb key.

    If `multiprocessing_read` is True, it will read all the images to memory
    using multiprocessing. Thus, your server needs to have enough memory.

    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        n_thread (int): For multiprocessing.
        map_size (int | None): Map size for lmdb env. If None, use the
            estimated size from images. Default: None
    """

    assert len(img_path_list) == len(keys), ('img_path_list and keys should have the same length, '
                                             f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Totoal images: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    def multiprocessing_read_batch(idx):
        # read batch images to memory (multiprocessing)
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        pbar = tqdm.tqdm(total=mp_batch, unit='image', desc='Read')

        def callback(arg):
            """get the image data and update pbar."""
            key, dataset[key], shapes[key] = arg
            pbar.update(1)
            # pbar.set_description(f'Read {key}')

        pool = Pool(n_thread)
        for path, key in zip(img_path_list[idx:idx+mp_batch], keys[idx:idx+mp_batch]):
            pool.apply_async(read_img_worker, args=(osp.join(data_path, path), key, compress_level), callback=callback)
        pool.close()
        pool.join()
        pbar.close()
        return dataset, shapes

    # create lmdb environment
    if map_size is None:
        # obtain data size for one image
        img = cv2.imread(osp.join(data_path, img_path_list[0]), cv2.IMREAD_UNCHANGED)
        _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        data_size_per_img = img_byte.nbytes
        print('Data size per image is: ', data_size_per_img)
        data_size = data_size_per_img * len(img_path_list)
        map_size = data_size * 10

    env = lmdb.open(lmdb_path, map_size=map_size)

    # write data to lmdb
    pbar = tqdm.tqdm(total=len(img_path_list), unit='chunk', desc='Write')
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(1)
        # pbar.set_description(f'Write {key}')
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            if idx % mp_batch == 0:
                dataset, shapes = multiprocessing_read_batch(idx)
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = read_img_worker(osp.join(data_path, path), key, compress_level)
            h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        required=True,
        type=Path,
        help='Data root directory',
    )
    parser.add_argument(
        '-o',
        required=True,
        type=Path,
        help='Path to output .lmdb file',
    )
    parser.add_argument(
        '-p',
        required=True,
        choices=['JPEG', 'png'],
        help='Images format',
    )
    args = parser.parse_args()
    create_lmdb(folder_path=str(args.i), lmdb_path=str(args.o), suffix=args.p)
