import torch
import torch.nn as nn
from torchsummary import summary


class QuizNet(nn.Module):
    def __init__(self):
        super(QuizNet, self).__init__()

        self.x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 3),
                bias=False,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.x2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

        self.x3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.x4 = nn.Sequential(nn.MaxPool2d(2, 2))
        self.x5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.x6 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )
        self.x7 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )
        self.x8 = nn.Sequential(nn.MaxPool2d(2, 2))
        self.x9 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )

        self.x10 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.x11 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

        self.x12 = nn.AvgPool2d(8)
        self.x13 = nn.Linear(64, 10)

    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x1 + x2)
        x4 = self.x4(x1 + x2 + x3)
        x5 = self.x5(x4)
        x6 = self.x6(x4 + x5)
        x7 = self.x7(x4 + x5 + x6)
        x8 = self.x8(x5 + x6 + x7)
        x9 = self.x9(x8)
        x10 = self.x10(x8 + x9)
        x11 = self.x11(x8 + x9 + x10)
        x12 = self.x12(x11).view(-1, 64)
        x13 = self.x13(x12)
        return x13


if __name__ == "__main__":
    model = QuizNet().to("cuda")
    summary(model, input_size=(3, 32, 32))

