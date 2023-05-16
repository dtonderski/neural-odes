import torchvision
from torch.utils.data import DataLoader
from .dataset import get_mnist_dataset, get_fashionmnist_dataset

def get_dataloaders(noise_std: float, dataset_name: str = 'mnist', batch_size: int = 256):
    return (
        get_dataloader(noise_std, dataset_name, True, batch_size),
        get_dataloader(noise_std, dataset_name, False, batch_size)
        )

def get_dataloader(noise_std: float, dataset_name: str = 'mnist', train: bool = True, batch_size: int = 256):
    match dataset_name.lower():
        case 'mnist':
            dataset = get_mnist_dataset(train, noise_std)
        case "fashionmnist":
            dataset = get_fashionmnist_dataset(train, noise_std)
        case _:
            raise ValueError(f"Unknown dataset_name type {dataset_name}")

    return get_dataloader_from_dataset(dataset, batch_size)

def get_dataloader_from_dataset(dataset: torchvision.datasets.FashionMNIST, batch_size: int = 128) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )