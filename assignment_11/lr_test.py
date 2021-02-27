import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import CustomNet
from tqdm import tqdm
from functools import partial
import copy
import config

tqdm = partial(tqdm, leave=True, position=0)


class LRRangeTest:
    def __init__(self, max_lr, min_lr, model, epochs, criterion, trainloader):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.model = model
        self.epochs = epochs
        self.criterion = criterion
        self.trainloader = trainloader
        self.lr_ = []
        self.train_acc = []

    def range_lr_test(self):
        lr = self.min_lr
        step_size = (self.max_lr - self.min_lr) / self.epochs
        print(f"step size: {step_size}")

        for _ in range(0, self.epochs):
            model = copy.deepcopy(self.model)
            print(f"current_lr: {lr}")
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.05)

            lr += step_size

            model.train()
            correct = 0
            total = 0

            for _, (data, target) in enumerate(
                tqdm(self.trainloader, total=len(self.trainloader))
            ):
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                optimizer.zero_grad()
                out = model(data)
                loss = self.criterion(out, target)
                pred = out.argmax(dim=1, keepdim=True)
                loss.backward()
                optimizer.step()
                correct += pred.eq(target.view_as(pred)).sum().item()

                total += len(target)
            print(f"train_accuracy: {100 * correct / total}")

            self.train_acc.append(100.0 * correct / total)
            self.lr_.append(optimizer.param_groups[0]["lr"])

        print(f"best training accuracy: {max(self.train_acc)}")
        print(f"best lr: {self.lr_[self.train_acc.index(max(self.train_acc))]}")

    def plot(self):
        plt.plot(self.lr_, self.train_acc)
        plt.title("LR Test Plot")
        plt.xlabel("Learning Rate")
        plt.ylabel("Training Accuracy")
        plt.show()

