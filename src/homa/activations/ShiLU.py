import torch
import torch.nn as nn
from .ActivationFunction import ActivationFunction


class ShiLU(ActivationFunction):
    def __init__(
        self, num_features: int, dim: int = 1, init_a: float = 1.0, init_b: float = 0.0
    ):
        super().__init__()
        self.num_features = num_features
        self.dim = dim
        self.a = nn.Parameter(torch.full((num_features,), float(init_a)))
        self.b = nn.Parameter(torch.full((num_features,), float(init_b)))

    def _reshape_params(self, x: torch.Tensor):
        shape = [1] * x.ndim
        feat_dim = self.dim if self.dim >= 0 else (x.ndim + self.dim)
        shape[feat_dim] = self.num_features
        return self.a.view(*shape), self.b.view(*shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self._reshape_params(x)
        return torch.relu(x) * a + b
