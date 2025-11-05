import torch
from .ActivationFunction import ActivationFunction


class ShiftedReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.max(-1, x)
