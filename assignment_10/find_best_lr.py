from torch_lr_finder import LRFinder
import torch.nn as nn
import torch.optim as optim
import dataloader
import config
from model import ResNet18


def find_best_lr(
    use_val_loader=False, start_lr=1e-5, end_lr=0.1, step_mode="linear", num_iter=100
):
    net = ResNet18().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=start_lr, momentum=0.9)
    lr_finder = LRFinder(net, optimizer, criterion, device=config.DEVICE)
    train_loader, test_loader = dataloader.get_iterators()
    if not use_val_loader:
        lr_finder.range_test(
            train_loader,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter,
            step_mode=step_mode,
        )
    else:
        lr_finder.range_test(
            train_loader,
            val_loader=test_loader,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter,
            step_mode=step_mode,
        )
    lr_finder.plot(log_lr=False)
    lr_finder.reset()
    return lr_finder

