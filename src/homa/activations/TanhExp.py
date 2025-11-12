import torch
from .ActivationFunction import ActivationFunction


class TanhExp(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.tanh(torch.exp(x))
