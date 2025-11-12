import torch
from .ActivationFunction import ActivationFunction


class DiffELU(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.a = 0.3
        self.b = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = x * x.exp() - self.b * torch.exp(self.b * x)
        return torch.where(x >= 0, x, self.a * delta)
