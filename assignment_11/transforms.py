from albumentations.augmentations.transforms import PadIfNeeded
import torchvision.transforms as transforms
from numpy import array
import albumentations
from albumentations.pytorch.transforms import ToTensor


class TrainTransforms:
    def __init__(self):
        self.train_transforms = albumentations.Compose(
            [
                albumentations.PadIfNeeded(
                    min_height=36, min_width=36, always_apply=True
                ),
                albumentations.RandomCrop(height=32, width=32, always_apply=True),
                albumentations.HorizontalFlip(),
                # albumentations.Rotate(2),
                albumentations.Cutout(num_holes=2, p=0.6),
                albumentations.Normalize(
                    mean=[0.4914, 0.48216, 0.44653], std=[0.2023, 0.1994, 0.2010]
                ),
                ToTensor(),
            ]
        )

    def __call__(self, image):
        image = array(image)
        image = self.train_transforms(image=image)
        return image["image"]


class TestTransforms:
    def __init__(self):
        self.test_transforms = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean=[0.4914, 0.48216, 0.44653], std=[0.2023, 0.1994, 0.2010]
                ),
                ToTensor(),
            ]
        )

    def __call__(self, image):
        image = array(image)
        image = self.test_transforms(image=image)
        return image["image"]

