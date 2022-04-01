import cv2
import os
import torch
import numpy as np
from basicsr.data.data_util import paths_from_lmdb, paths_from_folder
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, tensor2img
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class JPEGOpenCVDataset(data.Dataset):
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
        super(JPEGOpenCVDataset, self).__init__()
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
        img_gt = imfrombytes(img_bytes, float32=False)
        img_gt = mod_crop(img_gt, scale)

        # generate lq image
        # downsample
        h, w = img_gt.shape[0:2]
        img_lq = cv2.resize(img_gt, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)

        # add JPEG compression
        qf = self._sample_qf()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qf]
        _, encimg = cv2.imencode('.jpg', img_lq, encode_param)
        img_lq = cv2.imdecode(encimg, 1).astype(np.float32) / 255.
        img_gt = img_gt.astype(np.float32) / 255.

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

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq,
                'gt': img_gt,
                'lq_path': gt_path,
                'gt_path': gt_path,
                'qf': qf,
                }

    def __len__(self):
        return len(self.paths)

    def create_lq(self, img_hq):
        qf = self.opt['qf']
        assert isinstance(qf, int), "'create_lq' does not support multiple QF values"
        img_lq = []
        for img in img_hq:
            img = tensor2img(img, rgb2bgr=True, out_type=np.uint8)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), qf]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img_lq.append(cv2.imdecode(encimg, 1))
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        img_lq = torch.stack(img_lq, dim=0) / 255.
        return img_lq
