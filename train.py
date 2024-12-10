import torch
import yaml
import os

from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from argparse import ArgumentParser
from typing import List

from src.model import load_from_args, FlowMatchModel
from src.data import MNISTDataset


def train(
        model: FlowMatchModel,
        dataloader: DataLoader,
        exp_dir: str
) -> None:
    num_epochs = 5

    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader)
    )

    accelerator = Accelerator()
    model, dataloader, opt, lr_scheduler = accelerator.prepare(
        model, dataloader, opt, lr_scheduler
    )

    model.train()
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1} / {num_epochs}")
        for i, (image, label) in enumerate(dataloader):
            opt.zero_grad()

            loss = model(image, label)

            accelerator.backward(loss)
            opt.step()
            lr_scheduler.step()

            if i % 100 == 0:
                print(f"{i} / {len(dataloader)} iters.\tLoss: {loss.item():.6f}")

        torch.save(
            accelerator.get_state_dict(model, unwrap=True),
            os.path.join(exp_dir, f"checkpoint_{epoch + 1:02}.pt")
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--p_uncond", type=float, default=0.1)

    subparsers = parser.add_subparsers(required=True, dest="arch")

    unet_parser = subparsers.add_parser("unet")
    unet_parser.add_argument("--d_t", type=int, default=256)
    unet_parser.add_argument("--dims", type=List[int], default=[64, 128, 256])
    unet_parser.add_argument("--depths", type=List[int], default=[2, 2, 3])

    vit_parser = subparsers.add_parser("vit")
    vit_parser.add_argument("--patch_size", type=int, default=2)
    vit_parser.add_argument("--max_rel_pos", type=int, default=8)
    vit_parser.add_argument("--d_model", type=int, default=256)
    vit_parser.add_argument("--n_layers", type=int, default=6)
    vit_parser.add_argument("--n_heads", type=int, default=8)

    args = parser.parse_args()

    exp_dir = os.path.join("experiments/", args.exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    with open(os.path.join(exp_dir, "config.yaml"), "w+") as f:
        yaml.dump(args.__dict__, f)

    dataset = MNISTDataset()

    model = load_from_args(args)

    dataloader = DataLoader(
        dataset,
        batch_size=48,
        shuffle=True,
        pin_memory=True
    )

    train(model, dataloader, exp_dir)
