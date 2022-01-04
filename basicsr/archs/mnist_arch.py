from torch import nn
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_input = 784
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 784),
            nn.LeakyReLU(0.2),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)