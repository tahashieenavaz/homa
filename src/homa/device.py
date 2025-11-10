import torch


def get_device():
    if torch.backends.mps.is_available():
        return mps()
    if torch.cuda.is_available():
        return cuda()
    return cpu()


def cpu():
    return torch.device("cpu")


def cuda():
    return torch.device("cuda")


def mps():
    return torch.device("mps")


def device():
    return get_device()


def move(*modules):
    for module in modules:
        module.to(get_device())
