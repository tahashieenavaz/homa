import torch
from .ActivationFunction import ActivationFunction


class SigmoidDerivative(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x) * torch.sigmoid(x).pow(2)
