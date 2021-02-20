from torch_lr_finder import LRFinder
import torch.nn as nn
import torch.optim as optim
import dataloader
import config
from model import ResNet18
from torch.optim.lr_scheduler import OneCycleLR


net = ResNet18().to(config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
lr_finder = LRFinder(net, optimizer, criterion, device=config.DEVICE)
train_loader, test_loader = dataloader.get_iterators()

lr_finder.range_test(train_loader, end_lr=1)
lr_finder.plot(log_lr=False)

# scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=20)
