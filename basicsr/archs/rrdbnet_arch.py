import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle, pad, unpad


class NoiseInjectionCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x, noise_std=1.0):
        batch, _, height, width = x.shape
        noise = x.new_empty(batch, 1, height, width).normal_() * noise_std
        return torch.cat((x, self.weight * noise), dim=1)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, noise_injection=False):
        super(ResidualDenseBlock, self).__init__()
        add_ch = 1 if noise_injection else 0
        self.conv1 = nn.Conv2d(add_ch + num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(add_ch + num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(add_ch + num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(add_ch + num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(add_ch + num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if noise_injection:
            self.ni_1 = NoiseInjectionCat()
            self.ni_2 = NoiseInjectionCat()
            self.ni_3 = NoiseInjectionCat()
            self.ni_4 = NoiseInjectionCat()
            self.ni_5 = NoiseInjectionCat()
        else:
            self.ni_1 = nn.Identity()
            self.ni_2 = nn.Identity()
            self.ni_3 = nn.Identity()
            self.ni_4 = nn.Identity()
            self.ni_5 = nn.Identity()

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

    def __init__(self, num_feat, num_grow_ch=32, noise_injection=False):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, noise_injection)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, noise_injection)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, noise_injection)

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
                 ):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        add_ch = 1 if (noise_injection and noise_injection_upsample) else 0
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch, noise_injection=noise_injection)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat + add_ch, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat + add_ch, num_feat, 3, 1, 1)
        self.ni_1 = NoiseInjectionCat() if (noise_injection and noise_injection_upsample) else nn.Identity()
        self.ni_2 = NoiseInjectionCat() if (noise_injection and noise_injection_upsample) else nn.Identity()
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # We need to pad images for two reasons:
        # (1) We assume x is divisible by 4 in both spatial dimensions in `pixel_unshuffle`
        # (2) JPEG pads images to multiple of 16 before compressing, if we don't pad too, we will work in an offset
        # Hence we pad the images to multiple of 16
        x, padding = pad(x, padding=16)

        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(self.ni_1(F.interpolate(feat, scale_factor=2, mode='nearest'))))
        feat = self.lrelu(self.conv_up2(self.ni_2(F.interpolate(feat, scale_factor=2, mode='nearest'))))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        # crop padding if needed
        out = unpad(out, padding)

        return out
