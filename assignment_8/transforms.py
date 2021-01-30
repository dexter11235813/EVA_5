import torchvision.transforms as transforms

train_transforms = transforms.Compose(
    [
        transforms.ColorJitter(0.25, 0.2, 0.3),
        transforms.RandomRotation(2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159]
        ),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159]
        ),
    ]
)
