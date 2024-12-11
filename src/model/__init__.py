import torch
import torch.nn.functional as F
import yaml

from torch import nn
from collections import namedtuple
from tqdm import tqdm
from abc import ABC, abstractmethod

from .ConvNeXt import ClassConditionalConvNeXtFiLMUnet
from .util import SinusoidalPosEmb


class FlowMatchModel(nn.Module, ABC):
    def __init__(self, in_channels, num_classes, sigma_min=1e-8, p_uncond=0.1):
        super(FlowMatchModel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sigma_min = sigma_min
        self.p_uncond = p_uncond
        self.sigma_offset = 1 - sigma_min

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
    def sample(self, label, num_steps=50, guidance_scale=2.5, step="euler", image_size=28):
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
            elif step == "midpoint":
                t = t.view(-1, 1, 1, 1)
                h = 1 / num_steps
                x = x + h * self.guided_flow(x + 0.5 * h * flow, t + 0.5 * h, label, guidance_scale)
            else:
                raise NotImplementedError

        return x

    def forward(self, x, label):
        B = x.shape[0]

        label[torch.rand(B, device=label.device) < self.p_uncond] = self.num_classes

        t = torch.rand(B, device=x.device)
        t = t.view(-1, 1, 1, 1)

        x_0 = torch.randn_like(x)
        x_t = (1 - self.sigma_offset * t) * x_0 + t * x

        pred = self.pred_flow(x_t, t, label)

        return F.mse_loss(pred, x - self.sigma_offset * x_0)


class LatentFlowMatchWrapper(nn.Module):
    def __init__(
            self,
            encoder: VAEEncoder,
            decoder: VAEDecoder,
            model: FlowMatchModel,
            latent_scale: float = 1.0,
            sample: bool = False
    ):
        super(LatentFlowMatchWrapper, self).__init__()
        self.latent_scale = latent_scale
        self.sample = sample

        self.encoder = encoder
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        self.decoder = decoder
        self.decoder.eval()
        self.decoder.requires_grad_(False)

        self.model = model

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)

    @torch.inference_mode()
    def sample(self, *args, **kwargs):
        z = self.model.sample(*args, **kwargs) / self.latent_scale

        return self.decoder(z)

    def forward(self, x, label):
        dist = self.encoder(x)

        if self.sample:
            z = dist.sample()
        else:
            z = dist.mean

        z = z * self.latent_scale

        return self.model(x, label)


class UNetFlowMatchModel(FlowMatchModel):
    def __init__(
            self, in_channels, num_classes, d_t=256, dims=None, depths=None,
            sigma_min=1e-8, p_uncond=0.1
    ):
        super(UNetFlowMatchModel, self).__init__(in_channels, num_classes, sigma_min, p_uncond)
        self.in_channels = in_channels

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


def load_from_args(args) -> FlowMatchModel:
    if args.arch == "unet":
        return UNetFlowMatchModel(
            in_channels=1,
            num_classes=10,
            d_t=args.d_t,
            dims=args.dims,
            depths=args.depths,
            p_uncond=args.p_uncond
        )
    elif args.arch == "config":
        return load_from_config(args.config)
    else:
        raise NotImplementedError


def load_from_config(config_path: str) -> FlowMatchModel:
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = namedtuple("Args", config.keys())(*config.values())

    return load_from_args(args)
