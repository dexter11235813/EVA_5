import torch
import torch.nn as nn
import torch.nn.functional as F
import config

from tqdm import tqdm
from functools import partial

tqdm = partial(tqdm, leave=True, position=0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class Record:
    def __init__(self, train_acc, train_loss, test_acc, test_loss):
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.test_acc = test_acc
        self.test_loss = test_loss


class Trainer:
    def __init__(self, model):
        self.model = model
        self.train_acc = []
        self.train_loss = []
        self.test_acc = []
        self.test_loss = []

    def train(
        self,
        epochs,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        scheduler=None,
        device=config.DEVICE,
    ):
        for epoch in range(epochs):
            print(f"{epoch + 1} / {epochs}")

            self._train(train_loader, optimizer, device, loss_fn)
            test_loss = self._evaluate(test_loader, loss_fn)
            if scheduler:
                scheduler.step(test_loss)

        return Record(self.train_acc, self.train_loss, self.test_acc, self.test_loss)

    def _train(self, train_loader, optimizer, device, loss_fn):
        self.model.train()
        correct = 0
        train_loss = 0

        for _, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            optimizer.zero_grad()

            output = self.model(data)
            loss = loss_fn(output, target)

            train_loss += loss.detach()
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        self.train_loss.append(train_loss * 1.0 / len(train_loader.dataset))
        self.train_acc.append(100.0 * correct / len(train_loader.dataset))

        print(
            f" Training loss = {train_loss * 1.0 / len(train_loader.dataset)}, Training Accuracy : {100.0 * correct / len(train_loader.dataset)}"
        )

    def _evaluate(self, test_loader, loss_fn):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for _, (data, target) in tqdm(
                enumerate(test_loader), total=len(test_loader)
            ):
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                output = F.log_softmax(self.model(data), dim=1)
                test_loss += torch.nn.functional.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset) * 1.0
        self.test_loss.append(test_loss)
        self.test_acc.append(100.0 * correct / len(test_loader.dataset))

        print(
            f" Test loss = {test_loss}, Test Accuracy : {100.0 * correct / len(test_loader.dataset)}"
        )
        return test_loss


class Trial:
    def __init__(self, name, model, args):
        self.name = name
        self.model = model
        self.args = args
        self.Record = Record
        self.Trainer = Trainer(model)

    def run(self):
        self.Record = self.Trainer.train(**self.args)
