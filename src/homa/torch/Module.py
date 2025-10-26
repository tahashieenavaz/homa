import torch
from ..device import get_device


class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.to(get_device())
