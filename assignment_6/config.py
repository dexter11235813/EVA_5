import torch

torch.manual_seed(1)


DEVICE = "cuda"
BATCH_SIZE = 128
INIT_LR = 0.03
EPOCHS = 15
KWARGS = {"num_workers": 1, "pin_memory": True} if DEVICE is "cuda" else {}

