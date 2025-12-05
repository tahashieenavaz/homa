import torch
from .ActivationFunction import ActivationFunction


class NLReLU(ActivationFunction):
    def __init__(self, beta: float = 1.05):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.beta * x.clamp(min=0) + 1)
