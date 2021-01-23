import torch.nn as nn
import torch.optim as optim
import data_loader
import config
from model import CIFARNet, Trainer, Trial
from torch.optim.lr_scheduler import OneCycleLR

if __name__ == "__main__":
    model = CIFARNet(
        first_layer_output_size=config.FIRST_LAYER_OUTPUT_SIZE,
        num_classes=config.NUM_CLASSES,
    ).to(config.DEVICE)
    criterion = nn.functional.nll_loss
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = OneCycleLR(optimizer, max_lr=0.5, total_steps=20)
    train_loader, test_loader = data_loader.get_iterators()

    run = Trial(
        name="first_run",
        model=model,
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

