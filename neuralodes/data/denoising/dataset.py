import torch
import torchvision
from torch.utils.data import Dataset

class DenoisingDataset(Dataset):
    # This is a dataset that adds gaussian noise to a given dataset.
    def __init__(self, dataset: Dataset, noise_std: float):
        self.dataset = dataset
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        # Add gausiann noise to the data
        x, _ = self.dataset[idx]
        return x + self.noise_std * torch.randn_like(x), x

def get_fashionmnist_dataset(train: bool, noise_std: float, root: str = "data/FashionMNIST") -> DenoisingDataset:
    ds = torchvision.datasets.FashionMNIST(
        root,
        train=train,
        download=True,
        transform=get_normalization(),
    )
    return DenoisingDataset(ds, noise_std)

def get_mnist_dataset(train: bool, noise_std: float, root: str = "data/MNIST") -> DenoisingDataset:
    ds = torchvision.datasets.MNIST(
        root,
        train=train,
        download=True,
        transform=get_normalization(),
    )
    
    return DenoisingDataset(ds, noise_std)

def get_normalization() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )