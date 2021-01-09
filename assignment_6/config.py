import torch

torch.manual_seed(1)


DEVICE = "cuda"
BATCH_SIZE = 128
INIT_LR = 0.03
EPOCHS = 2
KWARGS = {"num_workers": 1, "pin_memory": True} if DEVICE is "cuda" else {}

SUMMARY_FOR_RUNS = "./images/summary_plots_for_runs.png"

L1_LAMBDA = 5e-5
WEIGHT_DECAY = 1e-5
