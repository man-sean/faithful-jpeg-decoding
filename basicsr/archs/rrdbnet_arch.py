from copy import deepcopy

import torch
import torchvision
from basicsr.utils.diffjpeg import CompressJpeg, DiffJPEG, DeCompressJpeg
from torch import nn as nn
from torch.nn import functional as F
from torch.profiler import record_function

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle, pad, unpad, freeze_module


class NoiseInjectionCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x, noise_std=1.0):
        batch, _, height, width = x.shape
        noise = x.new_empty(batch, 1, height, width).normal_() * noise_std
        return torch.cat((x, self.weight * noise), dim=1)


class NoiseInjectionAdd(nn.Module):
    def __init__(self):
        super().__init__()
        # self.weight = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.weight = nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x, noise_std=1.0):
        batch, _, height, width = x.shape
        noise = x.new_empty(batch, 1, height, width).normal_() * noise_std * self.weight
        return x + noise


noise_factory = {
    'cat': NoiseInjectionCat,
    'add': NoiseInjectionAdd,
}


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, noise_injection=False, noise_type='cat'):
        super(ResidualDenseBlock, self).__init__()
        add_ch = 1 if (noise_injection and noise_type in ['cat']) else 0
        self.conv1 = nn.Conv2d(add_ch + num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(add_ch + num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(add_ch + num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(add_ch + num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(add_ch + num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.ni_1 = noise_factory[noise_type]() if noise_injection else nn.Identity()
        self.ni_2 = noise_factory[noise_type]() if noise_injection else nn.Identity()
        self.ni_3 = noise_factory[noise_type]() if noise_injection else nn.Identity()
        self.ni_4 = noise_factory[noise_type]() if noise_injection else nn.Identity()
        self.ni_5 = noise_factory[noise_type]() if noise_injection else nn.Identity()

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(self.ni_1(x)))
        x2 = self.lrelu(self.conv2(self.ni_2(torch.cat((x, x1), 1))))
        x3 = self.lrelu(self.conv3(self.ni_3(torch.cat((x, x1, x2), 1))))
        x4 = self.lrelu(self.conv4(self.ni_4(torch.cat((x, x1, x2, x3), 1))))
        x5 = self.conv5(self.ni_5(torch.cat((x, x1, x2, x3, x4), 1)))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32, noise_injection=False, noise_type='cat'):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, noise_injection, noise_type)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, noise_injection, noise_type)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, noise_injection, noise_type)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 scale=4,
                 num_feat=64,
                 num_block=23,
                 num_grow_ch=32,
                 noise_injection=False,
                 noise_injection_upsample=False,
                 enforce_consistency=False,
                 noise_type='cat',
                 ):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self._enforce_consistency = enforce_consistency
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        add_ch = 1 if (noise_injection and noise_injection_upsample and noise_type in ['cat']) else 0
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch,
                               noise_injection=noise_injection, noise_type=noise_type)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat + add_ch, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat + add_ch, num_feat, 3, 1, 1)
        self.ni_1 = noise_factory[noise_type]() if (noise_injection and noise_injection_upsample) else nn.Identity()
        self.ni_2 = noise_factory[noise_type]() if (noise_injection and noise_injection_upsample) else nn.Identity()
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def enforce_consistency(self, compressed, restored, qf):
        if not self._enforce_consistency:
            return restored

        factor = DiffJPEG.quality_to_factor(deepcopy(qf))
        compressor = CompressJpeg(rounding=lambda x: x).to(compressed.device)
        decompressor = DeCompressJpeg(rounding=lambda x: x).to(compressed.device)

        compressed, c_padding = pad(compressed, padding=16)
        restored, r_padding = pad(restored, padding=16)

        compressed_coeffs = compressor(compressed, factor=deepcopy(factor))
        restored_coeffs = compressor(restored, factor=deepcopy(factor))

        for c_coeff, r_coeff in zip(compressed_coeffs, restored_coeffs):
            diff = (r_coeff - c_coeff)
            cond = (diff.abs() > 0.5)
            r_coeff[cond] = c_coeff[cond] + diff[cond].sign() * 0.49
            if torch.any((c_coeff - r_coeff).abs() > 0.5 + 1e-8):
                cond = ((r_coeff - c_coeff).abs() > 0.5)
                print(f'Coefficient deviation!\n{(r_coeff[cond]-c_coeff[cond])=}')

        consistent = decompressor(y=restored_coeffs[0], cb=restored_coeffs[1], cr=restored_coeffs[2],
                                  imgh=restored.shape[-2], imgw=restored.shape[-1], factor=factor).contiguous()

        restored = unpad(restored, r_padding)
        consistent = unpad(consistent, r_padding)

        return restored, consistent

    def forward(self, x, qf):
        # torchvision.utils.save_image(x[0], "/home/sean.man/RRDBNet_input_img.png")
        # torchvision.utils.save_image(x[16], "/home/sean.man/RRDBNet_input_img_16.png")
        # print(f'{x.shape=} {x.dtype=}')
        # y = ((x[0]*255).type(torch.uint8) - (x[0]*255)).abs()
        # print(f'{y.mean()} {y.std()}')

        # We need to pad images for two reasons:
        # (1) We assume x is divisible by 4 in both spatial dimensions in `pixel_unshuffle`
        # (2) JPEG pads images to multiple of 16 before compressing, if we don't pad too, we will work in an offset
        # Hence we pad the images to multiple of 16
        with record_function('pad_input'):
            x_, padding = pad(x, padding=16)

        with record_function('unshuffle'):
            if self.scale == 2:
                feat = pixel_unshuffle(x_, scale=2)
            elif self.scale == 1:
                feat = pixel_unshuffle(x_, scale=4)
            else:
                feat = x_
        with record_function('first_conv'):
            feat = self.conv_first(feat)
        with record_function('body_conv'):
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat

        # upsample
        with record_function('upsample'):
            feat = self.lrelu(self.conv_up1(self.ni_1(F.interpolate(feat, scale_factor=2, mode='nearest'))))
            feat = self.lrelu(self.conv_up2(self.ni_2(F.interpolate(feat, scale_factor=2, mode='nearest'))))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        with record_function('unpad_output'):
            # crop padding if needed
            out = unpad(out, padding)
        # torchvision.utils.save_image(out[0], "/home/sean.man/RRDBNet_output_img.png")

        # enforce consistency if necessary
        with record_function('enforce_consistency'):
            out = self.enforce_consistency(x, out, qf)

        return out
