import torch
import torch.nn.functional as F

from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.model.vae import VAEEncoder, VAEDecoder
from src.data import MNISTDataset


def train(
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        dataloader: DataLoader,
        kl_weight: float = 0.5
) -> None:
    num_epochs = 5

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-5)
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader)
    )

    accelerator = Accelerator()
    encoder, decoder, dataloader, opt, lr_scheduler = accelerator.prepare(
        encoder, decoder, dataloader, opt, lr_scheduler
    )

    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1} / {num_epochs}")
        for i, (image, _) in enumerate(dataloader):
            opt.zero_grad()

            dist = encoder(image)
            z = dist.sample()
            recon = decoder(z)

            kl_loss = dist.kl.mean()
            recon_loss = F.mse_loss(recon, image)

            loss = recon_loss + kl_weight * kl_loss

            accelerator.backward(loss)
            opt.step()
            lr_scheduler.step()

            if i % 100 == 0:
                print(f"{i} / {len(dataloader)} iters.\t"
                      f"Recon Loss: {recon_loss.item():.4f}\t"
                      f"KL Loss: {kl_loss.item():.4f}")

        torch.save(
            {
                "encoder": accelerator.get_state_dict(encoder, unwrap=True),
                "decoder": accelerator.get_state_dict(decoder, unwrap=True)
            },
            f"vae_checkpoints/checkpoint_{epoch + 1:02}.pt"
        )


if __name__ == "__main__":
    dataset = MNISTDataset()

    encoder = VAEEncoder(
        in_channels=1
    )
    decoder = VAEDecoder(
        in_channels=1
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        pin_memory=True
    )

    train(encoder, decoder, dataloader)
