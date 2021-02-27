import torch
import torch.nn as nn
import torch.nn.functional as F
import config

from tqdm import tqdm
from functools import partial
from torchsummary import summary

tqdm = partial(tqdm, leave=True, position=0)


class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.PrepLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False,
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.residual3 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.flatten = nn.Conv2d(
            in_channels=512, out_channels=10, kernel_size=1, bias=False
        )
        self.final_pool = nn.MaxPool2d(4, 4)

    def forward(self, x):
        x = self.PrepLayer(x)
        x = self.layer1(x)
        r1 = self.residual1(x)
        x = x + r1
        x = self.layer2(x)
        x = self.layer3(x)
        r2 = self.residual3(x)
        x = x + r2
        x = self.final_pool(x)
        x = self.flatten(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class Record:
    def __init__(self, train_acc, train_loss, test_acc, test_loss, lr):
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.test_acc = test_acc
        self.test_loss = test_loss
        self.lr = lr


class Trainer:
    def __init__(self, model):
        self.model = model
        self.train_acc = []
        self.train_loss = []
        self.test_acc = []
        self.test_loss = []
        self.LR = []

    def train(
        self,
        epochs,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        scheduler=None,
        batch_scheduler=True,
    ):
        for epoch in range(epochs):
            print(f"{epoch + 1} / {epochs}")
            clr = optimizer.param_groups[0]["lr"]
            print(f"current_lr: {clr}")
            self.LR.append(clr)
            # if not batch_scheduler:
            #     print("passing scheduler inside _train...")
            #     self._train(train_loader, optimizer, loss_fn, scheduler)
            # else:

            self._train(train_loader, optimizer, loss_fn)
            test_loss = self._evaluate(test_loader, loss_fn)
            if scheduler:
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    scheduler.step(test_loss)
                # elif scheduler.__class__.__name__ == "OneCycleLR":
                #     print("scheduler update passed over at the epoch level")
                #     continue
                else:
                    print("updating scheduler...")
                    scheduler.step()

        return Record(
            self.train_acc, self.train_loss, self.test_acc, self.test_loss, self.LR
        )

    def _train(self, train_loader, optimizer, loss_fn, scheduler=None):
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
            if scheduler:
                scheduler.step()
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
                output = self.model(data)
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
