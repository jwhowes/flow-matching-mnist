import torch

from torch import nn
from dataclasses import dataclass

from .conv import ConvNeXtBlock


@dataclass
class DiagonalGaussian:
    mean: torch.FloatTensor
    log_var: torch.FloatTensor

    def sample(self):
        return torch.randn_like(self.mean) * (0.5 * self.log_var).exp() + self.mean

    @property
    def kl(self):
        return 0.5 * (
            self.mean.pow(2) + self.log_var.exp() - 1.0 - self.log_var
        ).mean((1, 2, 3))


class VAEEncoder(nn.Module):
    def __init__(self, in_channels, latent_channels=3, dims=None, depths=None):
        super(VAEEncoder, self).__init__()
        if depths is None:
            depths = [2, 2, 1]

        if dims is None:
            dims = [64, 128, 256]

        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=7, padding=3)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_blocks.append(nn.Sequential(*[
                ConvNeXtBlock(dims[i]) for _ in range(depths[i])
            ]))

            self.down_samples.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))

        self.mid_blocks = nn.Sequential(*[ConvNeXtBlock(dims[-1]) for _ in range(depths[-1])])

        self.head = nn.Conv2d(dims[-1], 2 * latent_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)

        for block, sample in zip(self.down_blocks, self.down_samples):
            x = block(x)
            x = sample(x)

        x = self.mid_blocks(x)

        mean, log_var = self.head(x).chunk(2, 1)
        return DiagonalGaussian(mean, log_var)


class VAEDecoder(nn.Module):
    def __init__(self, in_channels, latent_channels=3, dims=None, depths=None):
        super(VAEDecoder, self).__init__()
        if dims is None:
            dims = [256, 128, 64]

        if depths is None:
            depths = [2, 2, 2]

        self.stem = nn.Conv2d(latent_channels, dims[0], kernel_size=7, padding=3)

        self.mid_blocks = nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(depths[0])])

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(1, len(dims)):
            self.up_samples.append(nn.ConvTranspose2d(dims[i - 1], dims[i], kernel_size=2, stride=2))

            self.up_blocks.append(nn.Sequential(*[
                ConvNeXtBlock(dims[i]) for _ in range(depths[i])
            ]))

        self.head = nn.Conv2d(dims[-1], in_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.mid_blocks(x)

        for block, sample in zip(self.up_blocks, self.up_samples):
            x = sample(x)
            x = block(x)

        return self.head(x)


if __name__ == "__main__":
    encoder = VAEEncoder(1)
    decoder = VAEDecoder(1)

    x = torch.randn(4, 1, 28, 28)
    dist = encoder(x)

    z = dist.sample()
    pred = decoder(z)

    print(pred.shape)
