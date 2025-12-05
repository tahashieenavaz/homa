import torch
from .ActivationFunction import ActivationFunction


class EANAF(ActivationFunction):
    def __init__(self):
        super().__init__()

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 2)

    def h(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.g(self.h(x))
