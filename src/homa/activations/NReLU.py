import torch
from .ActivationFunction import ActivationFunction


class NReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        sigma = x.std(unbiased=False)
        a = torch.randn_like(x) * sigma
        return torch.where(x >= 0, x + a, 0)
