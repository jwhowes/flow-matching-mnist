from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    def __init__(self, split="train"):
        self.ds = load_dataset("ylecun/mnist", split=split)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,), std=(0.3081,)
            )
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = self.ds[idx]

        return self.transform(data["image"]), data["label"]
