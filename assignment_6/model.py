import torch.nn as nn
from torch.nn.functional import log_softmax
from GhostBatchNorm import GhostBatchNorm


class Net(nn.Module):
    def __init__(self, ghost_norm=False):
        super().__init__()
        self.mnist_classifier = nn.Sequential(
            nn.Conv2d(1, 8, 3, bias=False),  # RF = 3
            nn.ReLU(),
            nn.BatchNorm2d(8) if not ghost_norm else GhostBatchNorm(8, 2),
            nn.Dropout(0.05),
            nn.Conv2d(8, 16, 3, bias=False),  # RF = 5
            nn.ReLU(),
            nn.BatchNorm2d(16) if not ghost_norm else GhostBatchNorm(16, 2),
            nn.MaxPool2d(2),  # RF = 6
            nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3, bias=False),  # RF = 10
            nn.ReLU(),
            nn.BatchNorm2d(16) if not ghost_norm else GhostBatchNorm(16, 2),
            nn.Dropout(0.05),
            nn.Conv2d(16, 16, 3, bias=False),  # RF = 14
            nn.ReLU(),
            nn.BatchNorm2d(16) if not ghost_norm else GhostBatchNorm(16, 2),
            nn.MaxPool2d(2),  # RF = 16
            nn.Dropout(0.1),
            nn.Conv2d(16, 10, 3, bias=False),  # RF = 24
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        x = self.mnist_classifier(x)
        x = x.view(-1, 10)
        return log_softmax(x, dim=-1)
