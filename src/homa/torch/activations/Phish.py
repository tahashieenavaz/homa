import torch
from .ActivationFunction import ActivationFunction


class Phish(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.nn.functional.gelu(x)
        return x * torch.tanh(a)
