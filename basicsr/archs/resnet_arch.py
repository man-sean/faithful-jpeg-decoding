import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data.distributed
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ResNetGANStabilityDiscriminator(nn.Module):
    def __init__(self, size, embed_size=256, nfilter=64):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 32
        nf = self.nf = nfilter

        # Submodules
        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)

        self.resnet_0_0 = ResnetBlock(1*nf, 1*nf)
        self.resnet_0_1 = ResnetBlock(1*nf, 2*nf)

        self.resnet_1_0 = ResnetBlock(2*nf, 2*nf)
        self.resnet_1_1 = ResnetBlock(2*nf, 4*nf)

        self.resnet_2_0 = ResnetBlock(4*nf, 4*nf)
        self.resnet_2_1 = ResnetBlock(4*nf, 8*nf)

        self.resnet_3_0 = ResnetBlock(8*nf, 8*nf)
        self.resnet_3_1 = ResnetBlock(8*nf, 16*nf)

        self.resnet_4_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_4_1 = ResnetBlock(16*nf, 16*nf)

        self.resnet_5_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_5_1 = ResnetBlock(16*nf, 16*nf)

        self.fc = nn.Linear(16*nf*s0*s0, 1)

    def forward(self, x, *args, **kwargs):
        batch_size = x.size(0)

        out = self.conv_img(x)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        # out = out.view(batch_size, 16*self.nf*self.s0*self.s0)
        out = out.reshape(batch_size, -1)
        out = self.fc(actvn(out))

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class Bias(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, **kwargs):
        return x + self.bias


class ConvBiasNormLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            Bias(out_channels),
        )

    def forward(self, x):
        return self.layers(x)


@ARCH_REGISTRY.register()
class ConditionalResNetDiscriminator(ResNetGANStabilityDiscriminator):
    def __init__(self, size, embed_size=256, nfilter=64):
        super().__init__(size=size, embed_size=embed_size, nfilter=nfilter)
        self.x_conv = nn.Conv2d(3, 1 * self.nf, 3, padding=1)
        self.y_conv = nn.Sequential(
            ConvBiasNormLayer(3, 64, 3, 1),
            ConvBiasNormLayer(64, 64, 3, 1),
            ConvBiasNormLayer(64, 64, 3, 1),
            ConvBiasNormLayer(64, 64, 3, 1),
        )
        self.conv_img = nn.Conv2d(self.nf + 64, 1*self.nf, 3, padding=1)

    def forward(self, x, y, *args, **kwargs):  # noqa
        img = torch.cat([self.x_conv(x), self.y_conv(y)], dim=1)
        return super().forward(x=img)



def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out

# net = ResNetGANStabilityDiscriminator(size=128, nfilter=32)
# print(sum(p.numel() for p in net.parameters() if p.requires_grad))
