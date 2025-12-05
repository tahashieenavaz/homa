import torch
from .ActivationFunction import ActivationFunction


class SiELU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chi = x + 0.044715 * x.pow(3)
        delta = 2 * torch.sqrt(torch.tensor(2) / torch.pi) * chi
        return x * torch.sigmoid(delta)
