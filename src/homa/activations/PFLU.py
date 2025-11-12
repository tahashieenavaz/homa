import torch
from .ActivationFunction import ActivationFunction


class PFLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        delta = x / (torch.sqrt(1 + x.pow(2)))
        return x * 0.5 * (1 + delta)
