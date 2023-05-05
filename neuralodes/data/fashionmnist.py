import torchvision
from torch.utils.data import DataLoader

def get_normalization() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

def get_dataset(train: bool, root: str = "data/FashionMNIST") -> torchvision.datasets.FashionMNIST:
    return torchvision.datasets.FashionMNIST(
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

def get_dataloader_split(train: bool, batch_size: int = 128) -> DataLoader:
    return get_dataloader_from_dataset(get_dataset(train), batch_size)