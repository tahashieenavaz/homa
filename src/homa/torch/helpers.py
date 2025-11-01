import torch
from ..device import get_device


def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs).to(get_device())
