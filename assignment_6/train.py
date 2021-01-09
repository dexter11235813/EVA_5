from tqdm import tqdm
from functools import partial
import torch
import config
import utils

tqdm = partial(tqdm, position=0, leave=True)

# Contains train, evaluation code, as well as code for identifying misclassified images
class Trainer:
    def __init__(self, model):
        self.model = model
        self.train_acc = []
        self.train_loss = []
        self.test_acc = []
        self.test_loss = []

    def train(
        self,
        EPOCHS,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        device=config.DEVICE,
        l1_penalty=False,
    ):
        for epoch in range(EPOCHS):
            print(f"{epoch + 1} / {EPOCHS}")
            utils.adjust_lr(optimizer, epoch)
            self._train(train_loader, optimizer, device, loss_fn, l1_penalty)
            self._evaluate(test_loader, device, loss_fn)

        return utils.Record(
            self.train_acc, self.train_loss, self.test_acc, self.test_loss
        )

    def _train(self, train_loader, optimizer, device, loss_fn, l1_penalty):
        self.model.train()
        correct = 0
        train_loss = 0

        for _, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            optimizer.zero_grad()

            output = self.model(data)
            loss = loss_fn(output, target)
            if l1_penalty:
                l1 = 0
                for p in self.model.parameters():
                    l1 = l1 + p.abs().sum()
                loss = loss + config.L1_LAMBDA * l1

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

    def _evaluate(self, test_loader, device, loss_fn):
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

    def get_misclassified_imgs(self, test_loader, device):
        mis = []
        mis_pred = []
        mis_target = []

        self.model.eval()
        with torch.no_grad():
            for _, (data, target) in tqdm(
                enumerate(test_loader), total=len(test_loader)
            ):
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)

                # https://discuss.pytorch.org/t/viewing-misclassified-image-predictions/82410
                idx_miss = pred.eq(target.view_as(pred)) == False

                misclassified_target = target.view_as(pred)[idx_miss]
                missclassified_pred = pred[idx_miss]
                missclassified = data[idx_miss]

                mis_pred.append(missclassified_pred)
                mis_target.append(misclassified_target)

                mis.append(missclassified)

        mis = torch.cat(mis)
        mis_pred = torch.cat(mis_pred)
        mis_target = torch.cat(mis_target)

        return mis, mis_pred, mis_target


# Entry point to the Trainer class
class Trial:
    def __init__(self, name, model, args):
        self.name = name
        self.model = model
        self.Record = None
        self.Trainer = Trainer(model)
        self.args = args

    def run(self):
        print(
            f"Running trial: {self.name} with batch_size = {self.args['train_loader'].batch_size}"
        )
        self.Record = self.Trainer.train(**self.args)
