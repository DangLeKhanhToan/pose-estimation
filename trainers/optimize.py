import torch.optim as optim

def create_optimizer(model, cfg):
    opt = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 1e-4)

    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")
    return optimizer

def create_scheduler(optimizer, cfg):
    step_size = cfg.get("step_size", 50)
    gamma = cfg.get("gamma", 0.1)
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
