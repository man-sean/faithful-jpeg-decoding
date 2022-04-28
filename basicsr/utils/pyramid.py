import torch
from torch import nn

from basicsr.utils.resize_right import resize
from basicsr.utils.dct import DCT, iDCT, BlockSplitting, BlockMerging
from basicsr.utils.diffjpeg import YCbCr2RGBJpeg, RGB2YCbCrJpeg


def gauss_pyr(img, n_levels, scale_factor=0.5):
    return [resize(img, scale_factor ** p).clip(0, 1) for p in range(n_levels)]


def laplace_pyr(img, n_levels, scale_factor=0.5):
    img_pyr = gauss_pyr(img, n_levels, scale_factor)
    img_pyr = [x - resize(y, 1 / scale_factor) for (x, y) in zip(img_pyr[:-1], img_pyr[1:])] + [img_pyr[-1]]
    return img_pyr


def encode_dct(img, size):
    device = img.device
    color = RGB2YCbCrJpeg().to(device)
    block = BlockSplitting(k=size).to(device)
    dct = DCT(n=size).to(device)

    ycbcr = color(img * 255.)
    return (dct(block(ycbcr[..., k])) for k in range(3))


def decode_dct(y, cb, cr, h, w, size):
    device = y.device
    idct = iDCT(n=size).to(device)
    block = BlockMerging(k=size).to(device)
    color = YCbCr2RGBJpeg().to(device)

    ycbcr = (block(idct(k), h, w) for k in (y, cb, cr))
    rgb = color(torch.stack(list(ycbcr), dim=-1))
    return rgb / 255.


def low_pass_dct(coeff, level):
    return coeff[..., :level, :level].clone()


def band_pass_dct(coeff, level):
    new_coeff = coeff[..., :level, :level].clone()
    new_coeff[..., :level - 1, :level - 1] = 0
    return new_coeff


def expand_dct(img, level):
    y, cb, cr = encode_dct(img, size=level)
    h, w = img.shape[-2], img.shape[-1]
    h, w = (h // level) * (level + 1), (w // level) * (level + 1)
    ycbcr = []
    for coeff in (y, cb, cr):
        ycbcr += [torch.nn.functional.pad(coeff, (0, 1, 0, 1))]
    y, cb, cr = tuple(ycbcr)
    return decode_dct(y, cb, cr, h, w, size=level + 1)


def dct_pyr(img):
    y, cb, cr = encode_dct(img, size=8)
    img_pyr = []
    for level in range(8, 0, -1):
        y_, cb_, cr_ = (low_pass_dct(c, level) for c in [y, cb, cr])
        h, w = img.shape[-2], img.shape[-1]
        h, w = h - (h // 8) * (8 - level), w - (w // 8) * (8 - level)
        x = decode_dct(y_, cb_, cr_, h, w, size=level)
        img_pyr += [x]
        # img_pyr += [(x - x.min()) / (x.max() - x.min())]
    return img_pyr


def band_dct_pyr(img):
    y, cb, cr = encode_dct(img, size=8)
    img_pyr = []
    for level in range(8, 0, -1):
        y_, cb_, cr_ = (band_pass_dct(c, level) for c in [y, cb, cr])
        h, w = img.shape[-2], img.shape[-1]
        h, w = h - (h // 8) * (8 - level), w - (w // 8) * (8 - level)
        x = decode_dct(y_, cb_, cr_, h, w, size=level)
        img_pyr += [x]
        # img_pyr += [(x - x.min()) / (x.max() - x.min())]
    return img_pyr


class DCTPyramid(nn.Module):
    def __init__(self):
        super(DCTPyramid, self).__init__()

        # colors
        self.rgb2ycbcr = RGB2YCbCrJpeg()
        self.ycbcr2rgb = YCbCr2RGBJpeg()

        # blocks
        self.block_splitting = nn.ModuleList([BlockSplitting(k=k) for k in range(1, 9)])
        self.block_merging = nn.ModuleList([BlockMerging(k=k) for k in range(1, 9)])

        # dct
        self.dct = nn.ModuleList([DCT(n=n) for n in range(1, 9)])
        self.idct = nn.ModuleList([iDCT(n=n) for n in range(1, 9)])

        # disable learning for this module
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, img, size):
        ycbcr = self.rgb2ycbcr(img * 255.)
        return (self.dct[size - 1](self.block_splitting[size - 1](ycbcr[..., k])) for k in range(3))

    def decode(self, y, cb, cr, h, w, size):
        ycbcr = (self.block_merging[size - 1](self.idct[size - 1](k), h, w) for k in (y, cb, cr))
        rgb = self.ycbcr2rgb(torch.stack(list(ycbcr), dim=-1))
        return rgb / 255.

    @staticmethod
    def band_pass(coeff, level):
        new_coeff = coeff[..., :level, :level].clone()
        new_coeff[..., :level - 1, :level - 1] = 0
        return new_coeff

    def expand(self, img, level):
        y, cb, cr = self.encode(img, size=level)
        h, w = img.shape[-2], img.shape[-1]
        h, w = (h // level) * (level + 1), (w // level) * (level + 1)
        ycbcr = []
        for coeff in (y, cb, cr):
            ycbcr += [torch.nn.functional.pad(coeff, (0, 1, 0, 1))]
        y, cb, cr = tuple(ycbcr)
        return self.decode(y, cb, cr, h, w, size=level + 1)

    def build(self, img):
        y, cb, cr = self.encode(img, size=8)
        img_pyr = []
        for level in range(8, 0, -1):
            y_, cb_, cr_ = (self.band_pass(c, level) for c in [y, cb, cr])
            h, w = img.shape[-2], img.shape[-1]
            h, w = h - (h // 8) * (8 - level), w - (w // 8) * (8 - level)
            x = self.decode(y_, cb_, cr_, h, w, size=level)
            img_pyr += [x]
        return img_pyr
