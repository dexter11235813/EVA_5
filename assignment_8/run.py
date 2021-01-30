import torch.nn as nn
import torch.optim as optim
import dataloader
import config
from model import ResNet18, Trainer, Trial
from torch.optim.lr_scheduler import OneCycleLR

if __name__ == "__main__":
    net = ResNet18().to(config.DEVICE)
    criterion = nn.functional.nll_loss
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=20)
    train_loader, test_loader = dataloader.get_iterators()

    run = Trial(
        name="first_run",
        model=net,
        args={
            "epochs": config.EPOCH,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "optimizer": optimizer,
            "device": config.DEVICE,
            "loss_fn": criterion,
            "scheduler": scheduler,
        },
    )

    run.run()
    print("Done!")
