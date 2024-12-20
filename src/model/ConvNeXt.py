import torch

from torch import nn

from .util import FiLM, LayerNorm2d


class ConvNeXtBlock(nn.Module):
    def __init__(self, d_model, hidden_size=None, norm_eps=1e-8):
        super(ConvNeXtBlock, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.norm = LayerNorm2d(d_model, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, d_model, kernel_size=1)
        )

    def forward(self, x):
        residual = x

        x = self.dwconv(x)
        x = self.norm(x)
        return residual + self.ffn(x)


class ConvNeXtFiLMBlock(nn.Module):
    def __init__(self, d_model, d_t, hidden_size=None, norm_eps=1e-8):
        super(ConvNeXtFiLMBlock, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.norm = FiLM(d_model, d_t, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, d_model, kernel_size=1)
        )

    def forward(self, x, t):
        residual = x

        x = self.dwconv(x)
        x = self.norm(x, t)

        return residual + self.ffn(x)


class ClassConditionalConvNeXtFiLMBlock(nn.Module):
    def __init__(self, d_model, d_t, num_classes, hidden_size=None, norm_eps=1e-6):
        super(ClassConditionalConvNeXtFiLMBlock, self).__init__()
        self.label_emb = nn.Embedding(num_classes, d_model)

        self.block = ConvNeXtFiLMBlock(d_model, d_t, hidden_size=hidden_size, norm_eps=norm_eps)

    def forward(self, x, t, label):
        B = x.shape[0]
        return self.block(x + self.label_emb(label).view(B, -1, 1, 1), t)


class ClassConditionalConvNeXtFiLMUnet(nn.Module):
    def __init__(self, in_channels, num_classes, d_t, dims=(64, 128, 256), depths=(2, 2, 3)):
        super(ClassConditionalConvNeXtFiLMUnet, self).__init__()
        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=7, padding=3)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(dims) - 1):
            blocks = nn.ModuleList()
            blocks.append(ClassConditionalConvNeXtFiLMBlock(dims[i], d_t, num_classes))
            for block in range(depths[i] - 1):
                blocks.append(ConvNeXtFiLMBlock(dims[i], d_t))

            self.down_blocks.append(blocks)
            self.down_samples.append(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )

        self.mid_blocks = nn.ModuleList()
        self.mid_blocks.append(ClassConditionalConvNeXtFiLMBlock(
            dims[-1], d_t, num_classes
        ))
        for block in range(depths[-1] - 1):
            self.mid_blocks.append(ConvNeXtFiLMBlock(dims[-1], d_t))

        self.up_samples = nn.ModuleList()
        self.up_combines = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(len(dims) - 2, -1, -1):
            self.up_samples.append(nn.ConvTranspose2d(
                dims[i + 1], dims[i], kernel_size=2, stride=2
            ))
            self.up_combines.append(nn.Conv2d(
                2 * dims[i], dims[i],
                kernel_size=1
            ))

            blocks = nn.ModuleList()
            blocks.append(ClassConditionalConvNeXtFiLMBlock(dims[i], d_t, num_classes))
            for block in range(depths[i] - 1):
                blocks.append(ConvNeXtFiLMBlock(dims[i], d_t))

            self.up_blocks.append(blocks)

        self.head = nn.Conv2d(dims[0], in_channels, kernel_size=7, padding=3)

    def forward(self, x, t, label):
        x = self.stem(x)

        acts = []
        for down_blocks, down_sample in zip(self.down_blocks, self.down_samples):
            x = down_blocks[0](x, t, label)
            for block in down_blocks[1:]:
                x = block(x, t)

            acts.append(x)
            x = down_sample(x)
        x = self.mid_blocks[0](x, t, label)
        for block in self.mid_blocks[1:]:
            x = block(x, t)

        for up_blocks, up_sample, up_combine, act in zip(
            self.up_blocks, self.up_samples, self.up_combines, acts[::-1]
        ):
            x = up_combine(torch.concatenate((
                up_sample(x),
                act
            ), dim=1))

            x = up_blocks[0](x, t, label)
            for block in up_blocks[1:]:
                x = block(x, t)

        return self.head(x)
