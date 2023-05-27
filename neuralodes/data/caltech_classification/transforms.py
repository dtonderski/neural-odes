import torchvision

class ConvertTo3Channels(object):
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        else:
            return img

def get_train_transform() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            ConvertTo3Channels(),
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.AutoAugment(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

def get_test_transform() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            ConvertTo3Channels(),
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )