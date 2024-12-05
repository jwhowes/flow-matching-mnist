import torch
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from abc import ABC, abstractmethod

from .conv import ClassConditionalConvNeXtFiLMUnet
from .util import SinusoidalPosEmb


class FlowMatchModel(nn.Module, ABC):
    @abstractmethod
    def pred_flow(
            self,
            x: torch.FloatTensor,
            t: torch.FloatTensor,
            label: torch.LongTensor
    ) -> torch.FloatTensor:
        ...

    @torch.inference_mode()
    def guided_flow(self, x, t, label, guidance_scale=2.5):
        if guidance_scale > 1.0:
            flow_uncond, flow_cond = self.pred_flow(
                torch.concat([x] * 2),
                torch.concat([t] * 2),
                label).chunk(2)

            return (1 - guidance_scale) * flow_uncond + guidance_scale * flow_cond
        else:
            return self.pred_flow(x, t, label)

    @torch.inference_mode()
    def sample(self, label, num_steps=50, guidance_scale=2.5, step="stochastic", image_size=28):
        B = label.shape[0]
        x = torch.randn(B, self.in_channels, image_size, image_size, device=label.device)

        ts = torch.linspace(0, 1, num_steps, device=label.device).unsqueeze(1).repeat(1, B)

        if guidance_scale > 1.0:
            label = torch.concat((
                torch.full((B,), self.num_classes, device=label.device),
                label
            ), dim=0)

        for i, t in tqdm(enumerate(ts), total=num_steps):
            flow = self.guided_flow(x, t, label, guidance_scale)

            if step == "euler":
                x = x + (1 / num_steps) * flow
            elif step == "stochastic":
                x = x + (1 - t).view(-1, 1, 1, 1) * flow

                if i < num_steps - 1:
                    next_t = ts[i + 1].view(-1, 1, 1, 1)
                    x = (1 - self.sigma_offset * next_t) * torch.randn_like(x) + next_t * x
            elif step == "heun":
                if i < num_steps - 1:
                    next_x = x + (1 / num_steps) * flow
                    flow2 = self.guided_flow(next_x, ts[i + 1], label, guidance_scale)

                    x = x + 0.5 * (1 / num_steps) * (flow + flow2)
                else:
                    x = x + (1 / num_steps) * flow
            else:
                raise NotImplementedError

        return x

    def forward(self, x, label):
        B = x.shape[0]

        label[torch.rand(B, device=label.device) < self.p_uncond] = self.num_classes

        t = torch.rand(B, device=x.device)
        t = t.view(-1, 1, 1, 1)

        x_0 = torch.randn_like(x)
        psi_t = (1 - self.sigma_offset * t) * x_0 + t * x

        pred = self.pred_flow(psi_t, t, label)

        return F.mse_loss(pred, x - self.sigma_offset * x_0)


class UNetFlowMatchModel(FlowMatchModel):
    def __init__(
            self, in_channels, num_classes, d_t=256, dims=None, depths=None,
            sigma_min=1e-8, p_uncond=0.1
    ):
        super(UNetFlowMatchModel, self).__init__()
        self.in_channels = in_channels
        self.p_uncond = p_uncond
        self.num_classes = num_classes

        self.sigma_min = sigma_min
        self.sigma_offset = 1 - sigma_min

        self.t_model = nn.Sequential(
            SinusoidalPosEmb(d_t),
            nn.Linear(d_t, 4 * d_t),
            nn.GELU(),
            nn.Linear(4 * d_t, d_t)
        )

        self.unet = ClassConditionalConvNeXtFiLMUnet(
            in_channels, num_classes + 1, d_t, dims, depths
        )

    def pred_flow(self, x, t, label):
        t_emb = self.t_model(t)
        return self.unet(x, t_emb, label)
