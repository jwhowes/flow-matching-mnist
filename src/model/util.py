import torch

from torch import nn


class FiLM2d(nn.Module):
    def __init__(self, d_model, d_t, *args, **kwargs):
        super(FiLM2d, self).__init__()
        self.gamma = nn.Linear(d_t, d_model, bias=False)
        self.beta = nn.Linear(d_t, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model, *args, elementwise_affine=False, **kwargs)

    def forward(self, x, t):
        B = x.shape[0]
        g = self.gamma(t).view(B, -1, 1, 1)
        b = self.beta(t).view(B, -1, 1, 1)

        return g * self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + b


class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model, base=1e5):
        super(SinusoidalPosEmb, self).__init__()
        assert d_model % 2 == 0
        self.d_model = d_model

        self.register_buffer(
            "theta",
            1.0 / (base ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

    def forward(self, x):
        x = x.view(-1, 1).float() * self.theta

        return torch.stack((
            x.cos(),
            x.sin()
        ), dim=-1).view(-1, self.d_model)

