import cv2
import os
import torch
import numpy as np
from copy import deepcopy

from basicsr.archs.arch_util import pad
from basicsr.data.data_util import paths_from_lmdb, paths_from_folder
from torch.utils import data as data
from torchvision.transforms.functional import normalize, center_crop

from basicsr.data.degradations import add_jpg_compression
from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir, get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.diffjpeg import DiffJPEG, CompressJpeg, quality_to_factor, q_table, y_table, c_table


@DATASET_REGISTRY.register()
class JPEGDataset(data.Dataset):
    """Example dataset.

    1. Read GT image
    2. Generate LQ (Low Quality) image with cv2 bicubic downsampling and JPEG compression

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(JPEGDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        else:
            self.paths = paths_from_folder(self.gt_folder)

        # diffjpeg_version = opt.get('diffjpeg_version', 1)
        # if diffjpeg_version == 1:
        #     self.jpeger = DiffJPEG(differentiable=True)
        # elif diffjpeg_version == 2:
        #     self.jpeger = DiffJPEGv2(differentiable=True)
        # else:
        #     raise ValueError('diffjpeg_version must be in [1, 2]')
        # logger = get_root_logger()
        # logger.info(f'Using DiffJPEG version {diffjpeg_version}.')
        self.jpeger = DiffJPEG(differentiable=True)
        self.compressor = CompressJpeg(rounding=torch.round)

    def _sample_qf(self):
        qf = self.opt['qf']
        if isinstance(qf, list):
            if len(qf) == 1:
                qf = qf[0]
            elif len(qf) == 2:
                min_qf, max_qf = qf[0], qf[1]
                qf = int(np.floor(np.clip(np.random.exponential(max_qf / min_qf) + min_qf, min_qf, max_qf)))
            else:
                raise ValueError(f'qf should be an integer or a list with 2 elements, got {len(qf)} elements.')
        elif isinstance(qf, int):
            qf = qf
        else:
            raise ValueError(f'qf should be an integer or a list with 2 elements, got {type(qf)}.')
        return qf

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = mod_crop(img_gt, scale)

        # generate lq image
        # downsample
        h, w = img_gt.shape[0:2]
        img_lq = cv2.resize(img_gt, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # generate lq image
        # add JPEG compression
        qf = self._sample_qf()
        y, cb, cr = self.compressor(image=pad(img_lq.unsqueeze(0), padding=16)[0],
                                    factor=self.jpeger.quality_to_factor(qf))
        y, cb, cr = (
            y.squeeze(0).detach(),
            cb.squeeze(0).detach(),
            cr.squeeze(0).detach(),
        )
        img_lq = self.jpeger(img_lq.unsqueeze(0), quality=qf).squeeze(0).detach()

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # center crop
        if self.opt.get('center_crop', None):
            img_gt = center_crop(img_gt, self.opt['center_crop'])
            img_lq = center_crop(img_lq, self.opt['center_crop'])

        return {'lq': img_lq,
                'gt': img_gt,
                'lq_path': gt_path,
                'gt_path': gt_path,
                'qf': qf,
                'y': y,
                'cb': cb,
                'cr': cr,
                }

    def __len__(self):
        return len(self.paths)

    def create_lq(self, img_hq):
        qf = self.opt['qf']
        assert isinstance(qf, int), "'create_lq' does not support multiple QF values"
        img_lq = self.jpeger(img_hq, quality=qf)
        return img_lq
