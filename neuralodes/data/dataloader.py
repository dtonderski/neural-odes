import torchvision
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name: str = 'mnist', batch_size: int = 256):
    return (
        get_dataloader(dataset_name, True, batch_size),
        get_dataloader(dataset_name, False, batch_size)
        )

def get_dataloader(dataset_name: str = 'mnist', train: bool = True, batch_size: int = 256):
    match dataset_name.lower():
        case 'mnist':
            dataset = get_mnist_dataset(train)
        case "fashionmnist":
            dataset = get_fashionmnist_dataset(train)
        case _:
            raise ValueError(f"Unknown dataset_name type {dataset_name}")

    return get_dataloader_from_dataset(dataset, batch_size)

def get_fashionmnist_dataset(train: bool, root: str = "data/FashionMNIST") -> torchvision.datasets.FashionMNIST:
    return torchvision.datasets.FashionMNIST(
        root,
        train=train,
        download=True,
        transform=get_normalization(),
    )

def get_mnist_dataset(train: bool, root: str = "data/MNIST") -> torchvision.datasets.MNIST:
    return torchvision.datasets.MNIST(
        root,
        train=train,
        download=True,
        transform=get_normalization(),
    )

def get_dataloader_from_dataset(dataset: torchvision.datasets.FashionMNIST, batch_size: int = 128) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

def get_normalization() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
