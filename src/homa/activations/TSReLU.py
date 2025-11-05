import torch
from .ActivationFunction import ActivationFunction


class TSReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.sigmoid(x))
