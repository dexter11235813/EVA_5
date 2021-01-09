import utils
import train
import config
import mnist_model
import dataloader
import torch


def create_trial(
    name, loss_fn, model_use_gbn=False, weight_decay=False, l1_penalty=False
):
    model = mnist_model.Net(ghost_norm=model_use_gbn).to(config.DEVICE)
    optimizer = (
        torch.optim.Adam(
            model.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY
        )
        if weight_decay
        else torch.optim.Adam(model.parameters(), lr=config.INIT_LR)
    )

    experiment = train.Experiment(
        name=name,
        model=model,
        train_args={
            "EPOCHS": config.EPOCHS,
            "train_loader": dataloader.train_loader,
            "test_loader": dataloader.test_loader,
            "optimizer": optimizer,
            "device": config.DEVICE,
            "loss_fn": loss_fn,
            "l1_penalty": l1_penalty,
        },
    )
    return experiment


if __name__ == "__main__":
    trials = []
    trial_1 = create_trial(
        "L1_with_BN",
        torch.nn.functional.nll_loss,
        model_use_gbn=False,
        weight_decay=False,
        l1_penalty=True,
    )
    trial_2 = create_trial(
        "L2_with_BN",
        torch.nn.functional.nll_loss,
        model_use_gbn=False,
        weight_decay=True,
        l1_penalty=False,
    )
    trial_3 = create_trial(
        "L1_and_L2_with_BN",
        torch.nn.functional.nll_loss,
        model_use_gbn=False,
        weight_decay=True,
        l1_penalty=True,
    )
    trial_4 = create_trial(
        "GBN",
        torch.nn.functional.nll_loss,
        model_use_gbn=True,
        weight_decay=False,
        l1_penalty=False,
    )
    trial_5 = create_trial(
        "L1_and_L2_with_GBN",
        torch.nn.functional.nll_loss,
        model_use_gbn=True,
        weight_decay=True,
        l1_penalty=True,
    )
    trials.extend([trial_1, trial_2, trial_3, trial_4, trial_5])

    for trial in trials:
        trial.run()

    print("finished Run")
    utils.plot_misclassified(25, trial_4, dataloader.test_loader, device=config.DEVICE)
    utils.plot_curves_for_trials(*trials)
    print("Done!!")
