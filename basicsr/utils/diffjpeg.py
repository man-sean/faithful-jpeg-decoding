"""
Modified from:
- https://github.com/mlomnitz/DiffJPEG
- https://queuecumber.gitlab.io/torchjpeg/_modules/torchjpeg/quantization/ijg.html

For images not divisible by 8
https://dsp.stackexchange.com/questions/35339/jpeg-dct-padding/35343#35343
"""
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy
from basicsr.utils.diff_round import diff_round, diff_round_ord3

# ------------------------ utils ------------------------#
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32)#.T
y_table = nn.Parameter(torch.from_numpy(y_table))
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]])#.T
c_table = nn.Parameter(torch.from_numpy(c_table))


def quality_to_factor(quality: torch.FloatTensor):
    """ Calculate factor corresponding to quality

    Args:
        quality(float): Quality for jpeg compression.

    Returns:
        float: Compression factor.
    """
    quality = quality.clone()
    quality[quality <= 0] = 1
    quality[quality > 100] = 100

    indices_0_50 = quality < 50
    indices_50_100 = quality >= 50

    quality[indices_0_50] = torch.div(5000, quality[indices_0_50], rounding_mode='trunc')
    quality[indices_50_100] = torch.trunc(200 - quality[indices_50_100] * 2)

    return quality


def q_table(base_table, factor):
    b = factor.size(0)
    mat = base_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
    mat = torch.div((mat + 50), 100, rounding_mode='trunc')
    mat[mat <= 0] = 1
    mat[mat > 255] = 255
    return mat


# ------------------------ compression ------------------------#
class RGB2YCbCrJpeg(nn.Module):
    """ Converts RGB image to YCbCr
    """

    def __init__(self):
        super(RGB2YCbCrJpeg, self).__init__()
        matrix = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
                          dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        """
        Args:
            image(Tensor): batch x 3 x height x width

        Returns:
            Tensor: batch x height x width x 3
        """
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        return result.view(image.shape)


class ChromaSubsampling(nn.Module):
    """ Chroma subsampling on CbCr channels
    """

    def __init__(self):
        super(ChromaSubsampling, self).__init__()

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width x 3

        Returns:
            y(tensor): batch x height x width
            cb(tensor): batch x height/2 x width/2
            cr(tensor): batch x height/2 x width/2
        """
        image_2 = image.permute(0, 3, 1, 2).clone()
        cb = F.avg_pool2d(image_2[:, 1, :, :].unsqueeze(1), kernel_size=2, stride=(2, 2), count_include_pad=False)
        cr = F.avg_pool2d(image_2[:, 2, :, :].unsqueeze(1), kernel_size=2, stride=(2, 2), count_include_pad=False)
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class BlockSplitting(nn.Module):
    """ Splitting image into patches
    """

    def __init__(self):
        super(BlockSplitting, self).__init__()
        self.k = 8

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor:  batch x h*w/64 x h x w
        """
        height, _ = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class DCT8x8(nn.Module):
    """ Discrete Cosine Transformation
    """

    def __init__(self):
        super(DCT8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class YQuantize(nn.Module):
    """ JPEG Quantization for Y channel

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding):
        super(YQuantize, self).__init__()
        self.rounding = rounding
        self.y_table = deepcopy(y_table)

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            factor = torch.tensor([factor], dtype=torch.float)
        table = q_table(self.y_table, factor)
        image = image.float() / table
        image = self.rounding(image)
        return image


class CQuantize(nn.Module):
    """ JPEG Quantization for CbCr channels

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding):
        super(CQuantize, self).__init__()
        self.rounding = rounding
        self.c_table = deepcopy(c_table)

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            factor = torch.tensor([factor], dtype=torch.float)
        table = q_table(self.c_table, factor)
        image = image.float() / table
        image = self.rounding(image)
        return image


class CompressJpeg(nn.Module):
    """Full JPEG compression algorithm

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding=torch.round):
        super(CompressJpeg, self).__init__()
        self.l1 = nn.Sequential(RGB2YCbCrJpeg(), ChromaSubsampling())
        self.l2 = nn.Sequential(BlockSplitting(), DCT8x8())
        self.c_quantize = CQuantize(rounding=rounding)
        self.y_quantize = YQuantize(rounding=rounding)

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x 3 x height x width

        Returns:
            dict(tensor): Compressed tensor with batch x h*w/64 x 8 x 8.
        """
        y, cb, cr = self.l1(image * 255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp, factor=factor)
            else:
                comp = self.y_quantize(comp, factor=factor)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


# ------------------------ decompression ------------------------#


class YDequantize(nn.Module):
    """Dequantize Y channel
    """

    def __init__(self):
        super(YDequantize, self).__init__()
        self.y_table = deepcopy(y_table)

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            factor = torch.tensor([factor], dtype=torch.float)
        table = q_table(self.y_table, factor)
        out = image * table
        return out


class CDequantize(nn.Module):
    """Dequantize CbCr channel
    """

    def __init__(self):
        super(CDequantize, self).__init__()
        self.c_table = deepcopy(c_table)

    def forward(self, image, factor=1):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        if isinstance(factor, (int, float)):
            factor = torch.tensor([factor], dtype=torch.float)
        table = q_table(self.c_table, factor)
        out = image * table
        return out


class iDCT8x8(nn.Module):
    """Inverse discrete Cosine Transformation
    """

    def __init__(self):
        super(iDCT8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width

        Returns:
            Tensor: batch x height x width
        """
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class BlockMerging(nn.Module):
    """Merge patches into image
    """

    def __init__(self):
        super(BlockMerging, self).__init__()

    def forward(self, patches, height, width):
        """
        Args:
            patches(tensor) batch x height*width/64, height x width
            height(int)
            width(int)

        Returns:
            Tensor: batch x height x width
        """
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class ChromaUpsampling(nn.Module):
    """Upsample chroma layers
    """

    def __init__(self):
        super(ChromaUpsampling, self).__init__()

    def forward(self, y, cb, cr):
        """
        Args:
            y(tensor): y channel image
            cb(tensor): cb channel
            cr(tensor): cr channel

        Returns:
            Tensor: batch x height x width x 3
        """

        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)
        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class YCbCr2RGBJpeg(nn.Module):
    """Converts YCbCr image to RGB JPEG
    """

    def __init__(self):
        super(YCbCr2RGBJpeg, self).__init__()

        matrix = np.array([[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        """
        Args:
            image(tensor): batch x height x width x 3

        Returns:
            Tensor: batch x 3 x height x width
        """
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        return result.view(image.shape).permute(0, 3, 1, 2)


class DeCompressJpeg(nn.Module):
    """Full JPEG decompression algorithm

    Args:
        rounding(function): rounding function to use
    """

    def __init__(self, rounding=torch.round):
        super(DeCompressJpeg, self).__init__()
        self.c_dequantize = CDequantize()
        self.y_dequantize = YDequantize()
        self.idct = iDCT8x8()
        self.merging = BlockMerging()
        self.chroma = ChromaUpsampling()
        self.colors = YCbCr2RGBJpeg()

    def forward(self, y, cb, cr, imgh, imgw, factor=1):
        """
        Args:
            compressed(dict(tensor)): batch x h*w/64 x 8 x 8
            imgh(int)
            imgw(int)
            factor(float)

        Returns:
            Tensor: batch x 3 x height x width
        """
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k], factor=factor)
                height, width = int(imgh / 2), int(imgw / 2)
            else:
                comp = self.y_dequantize(components[k], factor=factor)
                height, width = imgh, imgw
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)

        image = torch.min(255 * torch.ones_like(image), torch.max(torch.zeros_like(image), image))
        return image / 255


# ------------------------ main DiffJPEG ------------------------ #


class DiffJPEG(nn.Module):
    """This JPEG algorithm result is slightly different from cv2.
    DiffJPEG supports batch processing.

    Args:
        differentiable(bool): If True, uses custom differentiable rounding function, if False, uses standard torch.round
    """

    def __init__(self, differentiable=True, order=1):
        super(DiffJPEG, self).__init__()
        if differentiable:
            # print(f'DiffJPEG using {order=}')
            if order == 1:
                rounding = diff_round
            elif order == 3:
                rounding = diff_round_ord3
            else:
                raise NotImplementedError(f"DiffJPEG does not support {order=}")
        else:
            rounding = torch.round

        self.compress = CompressJpeg(rounding=rounding)
        self.decompress = DeCompressJpeg(rounding=rounding)

    @staticmethod
    def quality_to_factor(quality):
        factor = deepcopy(quality)
        if isinstance(factor, (int, float)):
            factor = torch.tensor([factor], dtype=torch.float)
        elif torch.is_tensor(factor):
            factor = factor.float()
        else:
            raise NotImplementedError(f'quality_to_factor is not implemented for {type(factor)}')
        factor = quality_to_factor(factor)
        return factor

    @staticmethod
    def pad_input(x):
        h, w = x.size()[-2:]
        h_pad, w_pad = 0, 0
        # why should use 16
        if h % 16 != 0:
            h_pad = 16 - h % 16
        if w % 16 != 0:
            w_pad = 16 - w % 16
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        return x, h, w, h_pad, w_pad

    def forward(self, x, quality):
        """
        Args:
            x (Tensor): Input image, bchw, rgb, [0, 1]
            quality(float): Quality factor for jpeg compression scheme.
        """
        factor = self.quality_to_factor(quality)
        x, h, w, h_pad, w_pad = self.pad_input(x)
        y, cb, cr = self.compress(x, factor=factor)
        recovered = self.decompress(y, cb, cr, (h + h_pad), (w + w_pad), factor=factor)
        recovered = recovered[:, :, 0:h, 0:w]
        return recovered


if __name__ == '__main__':
    import cv2

    from basicsr.utils import img2tensor, tensor2img

    img_gt = cv2.imread('test.png') / 255.

    # -------------- cv2 -------------- #
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
    _, encimg = cv2.imencode('.jpg', img_gt * 255., encode_param)
    img_lq = np.float32(cv2.imdecode(encimg, 1))
    cv2.imwrite('cv2_JPEG_20.png', img_lq)

    # -------------- DiffJPEG -------------- #
    jpeger = DiffJPEG(differentiable=False).cuda()
    img_gt = img2tensor(img_gt)
    img_gt = torch.stack([img_gt, img_gt]).cuda()
    quality = img_gt.new_tensor([20, 40])
    out = jpeger(img_gt, quality=quality)

    cv2.imwrite('pt_JPEG_20.png', tensor2img(out[0]))
    cv2.imwrite('pt_JPEG_40.png', tensor2img(out[1]))
