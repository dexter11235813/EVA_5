from numpy import pi

class Record:
    def __init__(self, train_acc, train_loss, valid_acc, valid_loss):
        self.train_acc = train_acc 
        self.train_loss = train_loss
        self.valid_acc = valid_acc
        self.valid_loss = valid_loss
        
def adjust_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        init_lr = param_group["lr"]
    lr = max(round(init_lr * 1 / (1 + pi / 50 * epoch), 10), 0.0005)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

