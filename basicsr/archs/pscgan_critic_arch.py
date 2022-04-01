import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.pscgan_critic_util import ConvLayer, ResBlock, EqualLinear, ClassicResBlock


@ARCH_REGISTRY.register()
class PSCGANCritic(nn.Module):
    def __init__(self, channels, input_spatial_extent):
        super().__init__()
        self.in_channels = channels[:-1]
        self.out_channels = channels[1:]
        self.f_rgb = ConvLayer(3, self.in_channels[0], 1, norm_type='bias')
        self.blocks = []
        self.init_channels()
        for index, (in_chan, out_chan) in enumerate(zip(self.in_channels,
                                                        self.out_channels)):
            block = ResBlock(in_chan, out_chan, 'down')
            self.blocks.append(block)
        self.blocks = nn.Sequential(*self.blocks)

        self.final_conv = ConvLayer(self.out_channels[-1], self.out_channels[-1], 3, norm_type='bias')
        out_spatial_extent = input_spatial_extent // 2 ** (len(self.out_channels))
        in_channels = (out_spatial_extent ** 2) * self.out_channels[-1]
        self.linear = nn.Sequential(EqualLinear(in_channels, self.out_channels[-1], activation=True),
                                    EqualLinear(self.out_channels[-1], 1, activation=False))

    def init_channels(self):
        pass

    def forward(self, x, **kwargs):
        out = self.f_rgb(x)
        out = self.blocks(out)
        out = self.final_conv(out).view(x.shape[0], -1)
        return self.linear(out)


class YPreProcessCritic(PSCGANCritic):
    def __init__(self, channels, input_spatial_extent):
        super().__init__(channels, input_spatial_extent)
        self.y_conv = nn.Sequential(
            ConvLayer(3, 64, 3, norm_type='bias'),
            ClassicResBlock(64, 64),
            ClassicResBlock(64, 64),
            ClassicResBlock(64, 64),
            ConvLayer(64, 64, 3, norm_type='bias')
        )

    def init_channels(self):
        self.in_channels[0] += 64

    def forward(self, x, y, **kwargs):  # noqa
        out = torch.cat((self.f_rgb(x), self.y_conv(y)), dim=1)
        out = self.blocks(out)
        out = self.final_conv(out).view(x.shape[0], -1)
        return self.linear(out)


@ARCH_REGISTRY.register()
class ConditionalPSCGANCritic(YPreProcessCritic):
    def __init__(self, channels, input_spatial_extent):
        super().__init__(channels, input_spatial_extent)
        self.y_conv = nn.Sequential(
            ConvLayer(3, 64, 3, norm_type='bias'),
            ConvLayer(64, 64, 3, norm_type='bias'),
            ConvLayer(64, 64, 3, norm_type='bias'),
            ConvLayer(64, 64, 3, norm_type='bias'),
        )
