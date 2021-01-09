from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config

train_loader = DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomRotation((-7, 7), fill=(0,)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    **config.KWARGS
)


test_loader = DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    **config.KWARGS
)
