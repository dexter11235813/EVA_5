from tqdm import tqdm
from functools import partial
import torch
import config
import utils

tqdm = partial(tqdm, position=0, leave=True)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.train_acc = []
        self.train_loss = []
        self.valid_acc = []
        self.valid_loss = []

    def train(
        self,
        EPOCHS,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        device=config.DEVICE,
    ):
        for epoch in range(EPOCHS):
            print(f"{epoch + 1} / {EPOCHS}")
            self._train(train_loader, optimizer, device, loss_fn)
            self._evaluate(test_loader, device)

        return utils.Record(
            self.train_acc, self.train_loss, self.valid_acc, self.valid_loss
        )

    def _train(self, train_loader, optimizer, device, loss_fn):
        pass

    def _evaluate(self, test_loader, device):
        pass
