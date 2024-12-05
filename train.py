import torch

from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.model import UNetFlowMatchModel, FlowMatchModel
from src.data import MNISTDataset


def train(
        model: FlowMatchModel,
        dataloader: DataLoader
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
            f"checkpoints/checkpoint_{epoch + 1:02}.pt"
        )


if __name__ == "__main__":
    dataset = MNISTDataset()

    model = UNetFlowMatchModel(
        in_channels=1,
        num_classes=10,
        d_init=64
    )

    dataloader = DataLoader(
        dataset,
        batch_size=48,
        shuffle=True,
        pin_memory=True
    )

    train(model, dataloader)
