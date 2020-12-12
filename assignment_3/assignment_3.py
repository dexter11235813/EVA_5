from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, bias=False)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3, bias=False)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3, bias=False)
        self.conv5 = nn.Conv2d(20, 30, kernel_size=3, bias=False)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(30, 62, kernel_size=3, bias=False)
        self.GAP = nn.AvgPool2d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.max_pool(
            self.dropout(self.relu(self.conv2(self.dropout(self.relu(self.conv1(x))))))
        )
        x = self.dropout(self.relu(self.conv4(self.dropout(self.relu(self.conv3(x))))))
        x = self.relu(self.conv5(x))
        x = self.GAP(self.conv6(x))
        # x = x.view(-1, 10)
        # print(x.shape)
        return F.log_softmax(x)


train_set = datasets.EMNIST(
    root="./data",
    train=True,
    download=True,
    split="byclass",
    transform=transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    ),
)
test_set = datasets.EMNIST(
    root="./data",
    train=False,
    download=True,
    split="byclass",
    transform=transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    ),
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)

print(summary(model, input_size=(1, 28, 28)))
tqdm = partial(tqdm, leave=True, position=0)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for _, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.squeeze(), target)
        loss.backward()
        optimizer.step()


# pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output.squeeze(), target
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    print(f"EPOCH {epoch} / 10")
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
