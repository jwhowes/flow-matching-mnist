import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from math import sqrt

from .util import SinusoidalPosEmb


class TimeDependentAttention(nn.Module):
    def __init__(self, d_model, d_t, n_heads):
        super(TimeDependentAttention, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.scale = sqrt(d_model // n_heads)

        self.Wx_q = nn.Linear(d_model, d_model, bias=False)
        self.Wx_k = nn.Linear(d_model, d_model, bias=False)
        self.Wx_v = nn.Linear(d_model, d_model, bias=False)

        self.Wt_q = nn.Linear(d_t, d_model, bias=False)
        self.Wt_k = nn.Linear(d_t, d_model, bias=False)
        self.Wt_v = nn.Linear(d_t, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, t, rel_pos_bias=None):
        t = t.unsqueeze(1)
        q = rearrange(
            self.Wx_q(x) + self.Wt_q(t), "b l (n d) -> b n l d", n=self.n_heads
        )
        k = rearrange(
            self.Wx_k(x) + self.Wt_k(t), "b l (n d) -> b n l d", n=self.n_heads
        )
        v = rearrange(
            self.Wx_v(x) + self.Wt_v(t), "b l (n d) -> b n l d", n=self.n_heads
        )

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        return self.W_o(rearrange(
            F.softmax(attn, dim=-1) @ v, "b n l d -> b l (n d)"
        ))


class DiffiTBlock(nn.Module):
    def __init__(self, d_model, d_t, n_heads, num_groups=32, patch_size=2, norm_eps=1e-6):
        super(DiffiTBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.GroupNorm(num_groups, d_model, eps=norm_eps),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        )

        self.patch_size = patch_size
        self.patch_emb = nn.Linear(d_model * patch_size * patch_size, d_model)

        self.attn = TimeDependentAttention(d_model, d_t, n_heads)
        self.attn_norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.patch_head = nn.Linear(d_model, d_model * patch_size * patch_size)

    def forward(self, x, t, rel_pos_bias=None):
        B, _, H, W = x.shape

        residual = x

        x = self.patch_emb(rearrange(
            self.conv(x), "b c (h hp) (w wp) -> b (h w) (c hp wp)",
            hp=self.patch_size,
            wp=self.patch_size
        ))

        x = x + self.attn(self.attn_norm(x), t, rel_pos_bias=rel_pos_bias)
        x = x + self.ffn(self.ffn_norm(x))

        return residual + rearrange(
            self.patch_head(x), "b (h w) (c hp wp) -> b c (h hp) (w wp)",
            h=(H // self.patch_size),
            w=(W // self.patch_size),
            hp=self.patch_size,
            wp=self.patch_size
        )


class DiffiTUNet(nn.Module):
    def __init__(
            self, in_channels, num_classes,
            max_rel_pos=7, d_t=256, dims=(64, 128, 256), depths=(2, 2, 3),
            n_heads=(4, 8, 8), num_groups=32, patch_size=2
    ):
        super(DiffiTUNet, self).__init__()
        self.label_emb = nn.Embedding(num_classes, d_t)

        self.tokenizer = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)

        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_path.append(nn.ModuleList([
                DiffiTBlock(dims[i], d_t, n_heads[i], num_groups, patch_size) for _ in range(depths[i])
            ]))
            self.down_samples.append(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )

        self.mid_blocks = nn.ModuleList([
            DiffiTBlock(dims[-1], d_t, n_heads[-1], num_groups, patch_size) for _ in range(depths[-1])
        ])

        self.up_path = nn.ModuleList()
        self.up_combines = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(len(dims) - 2, -1, -1):
            self.up_samples.append(nn.ConvTranspose2d(
                dims[i + 1], dims[i], kernel_size=2, stride=2
            ))
            self.up_combines.append(nn.Conv2d(
                2 * dims[i], dims[i], kernel_size=1
            ))
            self.up_path.append([
                DiffiTBlock(dims[i], d_t, n_heads[i], num_groups, patch_size) for _ in range(depths[i])
            ])

        self.head = nn.Conv2d(dims[0], in_channels, kernel_size=1)

    def forward(self, x, t, label):
        x = self.stem(x)
        t = t + label

        acts = []
        for down_blocks, down_sample in zip(self.down_path, self.down_samples):
            for block in down_blocks:
                x = block(x, t)
            acts.append(x)
            x = down_sample(x)

        for block in self.mid_blocks:
            x = block(x, t)

        for up_blocks, up_sample, up_combine, act in zip(
            self.up_path, self.up_samples, self.up_combines, acts[::-1]
        ):
            x = up_combine(torch.concatenate((
                up_sample(x),
                act
            ), dim=1))

            for block in up_blocks:
                x = block(x, t)

        return self.head(x)
