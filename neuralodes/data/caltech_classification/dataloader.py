import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from .transforms import get_train_transform, get_test_transform


def get_dataloaders(batch_size: int = 256, random_seed: int = 42):
    ds = torchvision.datasets.Caltech101("data/Caltech101", download=True)
    n_train = int(len(ds) * 0.8)
    
    train_ds, test_ds = random_split(ds, [n_train, len(ds) - n_train], generator=torch.Generator().manual_seed(random_seed))
    train_ds.dataset.transform = get_train_transform()
    test_ds.dataset.transform = get_test_transform()
    
    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(test_ds, batch_size, shuffle=False)
        )
