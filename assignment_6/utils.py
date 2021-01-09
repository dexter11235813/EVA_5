from numpy import pi, floor, ceil, sqrt
import matplotlib.pyplot as plt
import torch
import config


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


# def l1_loss(y_pred, target):
#     loss = torch.nn.functional.nll_loss(y_pred, target)

#     l1 = 0
#     for p in model.parameters():
#         l1 = l1 + p.abs().sum()
#     loss = loss + 5e-4 * l1
#     return loss


def plot_misclassified(number, trial, test_loader, device, save_path="./images"):
    images, predicted, actual = trial.Trainer.get_misclassified_imgs(
        test_loader, device
    )
    nrows = int(floor(sqrt(number)))
    ncols = int(ceil(sqrt(number)))
    save_path = f"{save_path}/{trial.name}"
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 15))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j

            ax[i, j].set_title(
                f"Predicted: {predicted[index]},\nActual : {actual[index]}"
            )
            ax[i, j].axis("off")
            ax[i, j].imshow(images[index].cpu().numpy())  # , cmap="gray_r")

    fig.savefig(save_path, bbox_inches="tight")
    print(f"Misclassified plot for {trial.name} saved at {save_path}")


def plot_curves_for_trials(*trials):
    train_loss = [trial.Record.train_loss for trial in trials]
    train_acc = [trial.Record.train_acc for trial in trials]
    test_loss = [trial.Record.test_loss for trial in trials]
    test_acc = [trial.Record.test_acc for trial in trials]
    data = [train_loss, train_acc, test_loss, test_acc]
    legends = [trial.name for trial in trials]
    titles = ["Train loss", "Train accuracy", "Test loss", "Test accuracy"]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

    for i in range(2):
        for j in range(2):
            ind = i * 2 + j
            ax[i, j].set_title(titles[ind])

            for k, legend in enumerate(legends):
                ax[i, j].plot(data[ind][k], label=legend)

            ax[i, j].legend()

    fig.savefig(config.SUMMARY_FOR_RUNS, bbox_inches="tight")

