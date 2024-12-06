import torch
import torch.nn.functional as F

from torch import nn
from math import sqrt
from einops import rearrange

from .util import FiLM, SinusoidalPosEmb


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super(SwiGLU, self).__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.hidden_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.out(
            F.silu(self.gate_proj(x)) * self.hidden_proj(x)
        )


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

    def forward(self, x, rel_pos_bias=None, rel_pos_coords=None):
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias[:, :, rel_pos_coords]

        x = F.softmax(attn, dim=-1) @ v

        return self.W_o(
            rearrange(x, "b n l d -> b l (n d)")
        )


class FiLMTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim=None, norm_eps=1e-8):
        super(FiLMTransformerBlock, self).__init__()
        self.attn = Attention(d_model, n_heads)
        self.attn_norm = FiLM(d_model, d_model, conv=False, eps=norm_eps)

        self.ffn = SwiGLU(d_model, hidden_dim)
        self.ffn_norm = FiLM(d_model, d_model, conv=False, eps=norm_eps)

    def forward(self, x, t, rel_pos_bias=None, rel_pos_coords=None):
        x = x + self.attn(self.attn_norm(x, t), rel_pos_bias, rel_pos_coords)

        return x + self.ffn(self.ffn_norm(x, t))


class FiLMViT(nn.Module):
    def __init__(self, image_size, max_rel_pos, d_model, n_layers, n_heads):
        super(FiLMViT, self).__init__()

        self.rel_pos_bias = nn.Parameter(
            torch.empty(1, n_heads, 2 * max_rel_pos - 1)
        )
        nn.init.xavier_normal_(self.rel_pos_bias.data)

        rel_pos_coords = torch.stack(torch.meshgrid([
            torch.arange(image_size), torch.arange(image_size)
        ], indexing="ij")).flatten(1)
        rel_pos_coords = (rel_pos_coords.unsqueeze(1) - rel_pos_coords.unsqueeze(2)).sum(0).clamp(
            -max_rel_pos + 1, max_rel_pos - 1
        )
        self.register_buffer(
            "rel_pos_coords",
            F.pad(rel_pos_coords, (1, 0, 1, 0), value=max_rel_pos),
            persistent=False
        )

        self.t_model = nn.Sequential(
            SinusoidalPosEmb(d_model),
            SwiGLU(d_model)
        )

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(FiLMTransformerBlock(d_model, n_heads))

    def forward(self, x, t):
        L = x.shape[1]
        t_emb = self.t_model(t)

        for layer in self.layers:
            x = layer(x, t_emb, self.rel_pos_bias[:, :, :L], self.rel_pos_coords[:L, :L])

        return x
