import torchvision.transforms as transforms

train_transforms = transforms.Compose(
    [
        transforms.RandomRotation(2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.48216, 0.44653], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.48216, 0.44653], std=[0.2023, 0.1994, 0.2010]
        ),
    ]
)
