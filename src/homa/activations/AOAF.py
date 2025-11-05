import torch
from .ActivationFunction import ActivationFunction


class AOAF(ActivationFunction):
    def __init__(self, b: float = 0.17, c: float = 0.17, dim: int = 1):
        super().__init__()
        self.b = b
        self.c = c
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=self.dim, keepdim=True)
        return torch.relu(x - self.b * mu) + self.c * mu
