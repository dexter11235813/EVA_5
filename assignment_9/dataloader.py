import torch
import torchvision
from transforms import TrainTransforms, TestTransforms
import config


def get_iterators(batch_size=config.BATCH_SIZE):
    train_transforms = TrainTransforms()
    test_transforms = TestTransforms()
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=train_transforms
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=test_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.BATCH_SIZE, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    return train_loader, test_loader
