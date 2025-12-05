import torch
from .ActivationFunction import ActivationFunction


class SinSig(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sin(torch.pi / 2 * torch.sigmoid(x))
