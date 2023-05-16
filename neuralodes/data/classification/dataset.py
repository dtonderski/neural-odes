import torchvision

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


def get_normalization() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
