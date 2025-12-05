import torch
from .ActivationFunction import ActivationFunction


class SoftsignRReLU(ActivationFunction):
    def __init__(self, lower: float = 1 / 8, upper: float = 1 / 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lower = lower
        self.upper = upper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.empty_like(x).uniform_(self.lower, self.upper)
        return torch.where(
            x >= 0,
            torch.div(1, (1 + x).pow(2)) + x,
            torch.div(1, (1 + x).pow(2)) + a * x,
        )
