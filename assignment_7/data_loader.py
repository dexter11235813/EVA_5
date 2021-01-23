import torch
import torchvision
from transforms import transform
import config


def get_iterators(batch_size=config.BATCH_SIZE):
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.BATCH_SIZE, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    return train_loader, test_loader
