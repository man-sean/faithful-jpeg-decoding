import math
from copy import deepcopy
from functools import partial

import torch
from basicsr.utils.diffjpeg import CompressJpeg, DiffJPEG, DeCompressJpeg
from torch import nn as nn
from torch.profiler import record_function
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.pyramid import band_dct_pyr, expand_dct, DCTPyramid
from basicsr.utils.resize_right import resize
from .arch_util import default_init_weights, make_layer, pixel_unshuffle, pad, unpad
from .rrdbnet_arch import RRDB, NoiseInjectionAdd


# class ConvLayer(nn.Module):
#     """
#
#     Args:
#         in_channels (int): Channel number of the input.
#         out_channels (int): Channel number of the output.
#         kernel_size (int): Size of the convolving kernel.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: int = 3,
#                  padding: int = 1,
#                  bias: bool = True,
#                  activation: bool = True,
#                  noise_injection: bool = False,
#                  ):
#         super(ConvLayer, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size=(kernel_size, kernel_size),
#                               stride=(1, 1),
#                               padding=padding,
#                               bias=bias)
#         self.noise_injection = NoiseInjectionAdd() if noise_injection else nn.Identity()
#         self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True) if activation else nn.Identity()
#
#     def forward(self, x):
#         """Forward function.
#
#         Args:
#             x (Tensor): Tensor with shape (b, c, h, w).
#
#         Returns:
#             Tensor: tensor after convolution.
#         """
#         out = self.noise_injection(x)
#         out = self.conv(out)
#         out = self.activation(out)
#
#         return out
#
#
# class StyleConv(nn.Module):
#     """To RGB from features.
#
#     Args:
#         in_channels (int): Channel number of input.
#         out_channels (int): Channel number of output.
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, noise_injection):
#         super(StyleConv, self).__init__()
#         self.conv = ConvLayer(in_channels, out_channels, kernel_size=kernel_size,
#                               bias=True, activation=True, noise_injection=noise_injection)
#
#     def forward(self, x):
#         """Forward function.
#
#         Args:
#             x (Tensor): Feature tensor with shape (b, c, h, w).
#
#         Returns:
#             Tensor: RGB images.
#         """
#         out = self.conv(x)
#         return out
#
#
# class ToRGB(nn.Module):
#     """To RGB from features.
#
#     Args:
#         in_channels (int): Channel number of input.
#         out_channels (int): Channel number of output.
#     """
#
#     def __init__(self, in_channels, out_channels):
#         super(ToRGB, self).__init__()
#         self.conv = ConvLayer(in_channels, out_channels, kernel_size=1, padding=0,
#                               bias=True, activation=False, noise_injection=False)
#
#     def forward(self, x):
#         """Forward function.
#
#         Args:
#             x (Tensor): Feature tensor with shape (b, c, h, w).
#
#         Returns:
#             Tensor: RGB images.
#         """
#         out = self.conv(x)
#         return out


@ARCH_REGISTRY.register()
class MultiScaleNet(nn.Module):
    """Multiscale restoration network based on laplacian pyramid decomposition of its input.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 in_ch: int = 3,
                 num_feat: int = 64,
                 num_grow_ch: int = 32,
                 noise_injection: bool = False,
                 ):
        super(MultiScaleNet, self).__init__()

        self.dct_pyramid = DCTPyramid()

        self.first_from_rgb = nn.Conv2d(in_ch, num_feat, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)
        self.first_pre_conv = RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch,
                                   noise_injection=noise_injection, noise_type='add')
        self.first_to_rgb = nn.Conv2d(num_feat, in_ch, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)

        self.pre_convs = nn.ModuleList()
        self.pre_from_rgbs = nn.ModuleList()
        self.main_convs = nn.ModuleList()
        self.from_rgbs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        for level in range(7):
            self.pre_from_rgbs.append(
                nn.Conv2d(
                    in_ch,
                    num_feat,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                    bias=True
                ))
            self.pre_convs.append(
                RRDB(
                    num_feat=num_feat,
                    num_grow_ch=num_grow_ch,
                    noise_injection=noise_injection,
                    noise_type='add',
                ))
            self.main_convs.append(
                RRDB(
                    num_feat=num_feat,
                    num_grow_ch=num_grow_ch,
                    noise_injection=False,
                ))
            self.from_rgbs.append(
                nn.Conv2d(
                    in_ch,
                    num_feat,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                    bias=True
                ))
            self.to_rgbs.append(
                nn.Conv2d(
                    num_feat,
                    in_ch,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                    bias=True
                ))

    def forward(self, x, *args, **kwargs):
        # JPEG pads images to multiple of 16 before compressing,
        # if we don't pad too, we will work in an offset.
        # Hence, we pad the images to multiple of 16
        with record_function('pad_input'):
            x_, padding = pad(x, padding=16)

        with record_function('build_pyramid'):
            pyramid = list(reversed(self.dct_pyramid.build(x_)))
        with record_function('level_1'):
            out = pyramid[0]
            out = self.first_from_rgb(out)
            out = self.first_pre_conv(out)
            out = self.first_to_rgb(out)

        for level, (input, pre_from_rgb, pre_conv, from_rgb, main_conv, to_rgb) in enumerate(zip(
                pyramid[1:],
                self.pre_from_rgbs,
                self.pre_convs,
                self.from_rgbs,
                self.main_convs,
                self.to_rgbs,
        ), start=2):
            # preprocess current level (including noise_injection)
            with record_function(f'preprocess_level_{level}'):
                input = pre_from_rgb(input)
                input = pre_conv(input)

            # upsample previous levels
            with record_function(f'expand_level_{level}'):
                out = self.dct_pyramid.expand(out, level=level - 1)
                out = from_rgb(out)

            with record_function(f'postprocess_level_{level}'):
                # merge current level to previous levels
                out += input
                out = main_conv(out)

                # convert back to RGB
                out = to_rgb(out)

        # crop padding if needed
        with record_function('unpad_output'):
            out = unpad(out, padding)

        return out.contiguous()
