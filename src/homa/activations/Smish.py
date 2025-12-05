import torch
from .ActivationFunction import ActivationFunction


class Smish(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.tanh(torch.log1p(torch.sigmoid(x)))
