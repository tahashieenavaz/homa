import torch
from .ActivationFunction import ActivationFunction


class TBSReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = 1 - torch.exp(-x)
        b = 1 + torch.exp(-x)
        c = a / b
        return x * torch.tanh(c)
