import torch.nn as nn
import torch.nn.functional as F
import config
from tqdm import tqdm
from functools import partial
import torch
from torchsummary import summary

tqdm = partial(tqdm, position=0, leave=True)


def conv_block(
    num_layers,
    in_channel,
    kernel_size,
    padding,
    dialation=1,
    depthwise=False,
    dropout=config.DROPOUT_RATE,
):
    module_list = []
    for j in range(num_layers):
        if not depthwise:
            module_list.extend(
                [
                    nn.Conv2d(
                        in_channel,
                        in_channel * 2,
                        kernel_size=3,
                        dilation=dialation,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(in_channel * 2),
                ]
            )
        else:

            module_list.extend(
                [
                    DepthWiseSeparableConv(
                        in_channel, in_channel * 2, padding=padding,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(in_channel * 2),
                ]
            )
        if j != num_layers - 1:
            module_list.append(nn.Dropout(dropout))
        in_channel = in_channel * 2

    return nn.Sequential(*module_list), in_channel


def transition_block(in_channel, out_channel, other_layers=True):
    if out_channel > in_channel:
        raise ValueError(
            f"out_channels {out_channel} should be lower than in_channels {in_channel} for the 1x1 convolution"
        )
    output = []
    output.extend(
        [
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), bias=False),
        ]
    )

    if other_layers:
        output.extend(
            [nn.ReLU(), nn.BatchNorm2d(out_channel), nn.Dropout(config.DROPOUT_RATE)]
        )

    return nn.Sequential(*output)


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.depthwise_layer = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=padding,
            groups=int(out_channels / in_channels),
        )
        self.separable_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.separable_layer(self.depthwise_layer(x))


class CIFARNet(nn.Module):
    def __init__(self, first_layer_output_size, num_classes):
        super(CIFARNet, self).__init__()
        self.num_classes = num_classes
        self.first_block = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=first_layer_output_size,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(first_layer_output_size),
            nn.Dropout(config.DROPOUT_RATE),
        )
        self.conv_block1, final_out_conv1 = conv_block(
            num_layers=2,
            in_channel=first_layer_output_size,
            kernel_size=3,
            padding=1,
            depthwise=False,
        )
        self.trans_block1 = transition_block(
            in_channel=final_out_conv1, out_channel=first_layer_output_size
        )

        self.conv_block2, final_out_conv2 = conv_block(
            num_layers=2,
            in_channel=first_layer_output_size,
            kernel_size=3,
            padding=1,
            dialation=2,
            depthwise=False,
        )

        self.trans_block2 = transition_block(
            in_channel=final_out_conv2, out_channel=first_layer_output_size
        )

        self.conv_block3, final_out_conv_3 = conv_block(
            num_layers=2,
            in_channel=first_layer_output_size,
            kernel_size=3,
            padding=1,
            depthwise=True,
        )

        self.trans_block3 = transition_block(
            in_channel=final_out_conv_3, out_channel=num_classes, other_layers=False,
        )
        self.GAP = nn.AvgPool2d(3)

    def forward(self, x):

        x = self.first_block(x)
        x = self.trans_block1(self.conv_block1(x))
        x = self.trans_block2(self.conv_block2(x))
        x = self.trans_block3(self.conv_block3(x))
        x = self.GAP(x).view(-1, self.num_classes)
        return F.log_softmax(x, dim=1)


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
            self._evaluate(test_loader, loss_fn)
            if scheduler:
                scheduler.step()

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


class Trial:
    def __init__(self, name, model, args):
        self.name = name
        self.model = model
        self.args = args
        self.Record = Record
        self.Trainer = Trainer(model)

    def run(self):
        self.Record = self.Trainer.train(**self.args)
