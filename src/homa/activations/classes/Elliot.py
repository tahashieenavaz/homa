import torch
from .ActivationFunction import ActivationFunction


class Elliot(ActivationFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 + torch.div(0.5 * x, 1 + torch.abs(x))
