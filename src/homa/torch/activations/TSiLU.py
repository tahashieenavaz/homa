import torch
from .ActivationFunction import ActivationFunction


class TSiLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = x / (1 + torch.exp(-x))
        beta = torch.exp(alpha) - torch.exp(-alpha)
        theta = torch.exp(alpha) + torch.exp(alpha)
        return beta / theta
