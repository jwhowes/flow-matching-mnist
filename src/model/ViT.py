import torch
import torch.nn.functional as F

from torch import nn
from math import sqrt
from einops import rearrange

from .util import FiLM, SinusoidalPosEmb


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.scale = sqrt(d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        x = F.softmax(attn, dim=-1) @ v

        return self.W_o(
            rearrange(x, "b n l d -> b l (n d)")
        )


class FiLMTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim=None, norm_eps=1e-8):
        super(FiLMTransformerBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model
        self.attn = Attention(d_model, n_heads)
        self.attn_norm = FiLM(d_model, d_model, conv=False, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.ffn_norm = FiLM(d_model, d_model, conv=False, eps=norm_eps)

    def forward(self, x, t):
        x = x + self.attn(self.attn_norm(x, t))

        return x + self.ffn(self.ffn_norm(x, t))


class FiLMTransformer(nn.Module):
    def __init__(self, length, d_model, n_layers, n_heads):
        super(FiLMTransformer, self).__init__()

        self.pos_emb = nn.Parameter(
            torch.empty(1, length, d_model).uniform_(
                -1.0 / sqrt(d_model), 1.0 / sqrt(d_model)
            )
        )

        self.t_model = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(FiLMTransformerBlock(d_model, n_heads))

    def forward(self, x, t):
        x = x + self.pos_emb
        t_emb = self.t_model(t)

        for layer in self.layers:
            x = layer(x, t_emb)

        return x
