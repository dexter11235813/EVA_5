import torch

torch.manual_seed(1)


DEVICE = "cuda"
BATCH_SIZE_BN = 128
BATCH_SIZE_GBN = 64
INIT_LR = 0.03
EPOCHS = 25
KWARGS = {"num_workers": 1, "pin_memory": True} if DEVICE is "cuda" else {}

SUMMARY_LOSS = "./images/summary_plot_for_runs_loss.png"
SUMMARY_ACC = "./images/summary_plot_for_runs_acc.png"

L1_LAMBDA = 5e-5
WEIGHT_DECAY = 1e-5
