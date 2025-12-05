import torch
from .ActivationFunction import ActivationFunction


class RReLU(ActivationFunction):
    def __init__(self, lower: int = 3, upper: int = 8, training: bool = False):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.training = training

    def forward(self, x):
        if self.training:
            a = torch.empty_like(x).uniform_(self.lower, self.upper)
        else:
            a = (self.lower + self.upper) / 2.0
        return torch.where(x >= 0, x, x / a)
