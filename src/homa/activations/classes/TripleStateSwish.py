import torch
from .ActivationFunction import ActivationFunction


class TripleStateSwish(ActivationFunction):
    def __init__(self, alpha: float = 20, beta: float = 40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = 1 / (1 + torch.exp(-x))
        b = 1 / (1 + torch.exp(-x + self.alpha))
        c = 1 / (1 + torch.exp(-x + self.beta))
        return x * a * (a + b + c)
