from numpy import pi, floor, ceil, sqrt
import matplotlib.pyplot as plt
import torch
import config
import sys


class Record:
    def __init__(self, train_acc, train_loss, test_acc, test_loss):
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.test_acc = test_acc
        self.test_loss = test_loss


def adjust_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        init_lr = param_group["lr"]
    lr = max(round(init_lr * 1 / (1 + pi / 50 * epoch), 10), 0.0005)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_misclassified(number, trial, test_loader, device, save_path="./images"):
    images, predicted, actual = trial.Trainer.get_misclassified_imgs(
        test_loader, device
    )
    nrows = int(floor(sqrt(number)))
    ncols = int(ceil(sqrt(number)))
    save_path = f"{save_path}/{trial.name}"
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 15))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j

            ax[i, j].set_title(
                f"Predicted: {predicted[index]},\nActual : {actual[index]}"
            )
            ax[i, j].axis("off")
            ax[i, j].imshow(images[index].cpu().numpy())

    fig.savefig(save_path, bbox_inches="tight")
    print(f"Misclassified plot for {trial.name} saved at {save_path}")


def plot_curves_for_trials(*trials):
    train_loss = [trial.Record.train_loss for trial in trials]
    train_acc = [trial.Record.train_acc for trial in trials]
    test_loss = [trial.Record.test_loss for trial in trials]
    test_acc = [trial.Record.test_acc for trial in trials]
    data_acc = [train_acc, test_acc]
    data_loss = [train_loss, test_loss]

    legends = [trial.name for trial in trials]
    titles_loss = ["Train loss", "Test loss"]
    titles_acc = ["Train accuracy", "Test accuracy"]

    fig1, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))
    fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))

    for i in range(2):
        ax1[i].set_title(titles_loss[i])

        for k, legend in enumerate(legends):
            ax1[i].plot(data_loss[i][k], label=legend)

        ax1[i].legend()

    for j in range(2):
        ax2[j].set_title(titles_acc[j])

        for k, legend in enumerate(legends):
            ax2[j].plot(data_acc[j][k], label=legend)

        ax2[j].legend()

    fig1.savefig(config.SUMMARY_LOSS, bbox_inches="tight")
    fig2.savefig(config.SUMMARY_ACC, bbox_inches="tight")

