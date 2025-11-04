import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GALU(nn.Module):
    def __init__(self, channels: int, max_input: float = 1.0):
        super().__init__()
        if max_input <= 0:
            raise ValueError("max_input must be positive.")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.channels = int(channels)
        self.max_input = float(max_input)
        self.alpha = Parameter(torch.empty(self.channels))
        self.beta = Parameter(torch.empty(self.channels))
        self.gamma = Parameter(torch.empty(self.channels))
        self.delta = Parameter(torch.empty(self.channels))
        self.reset_parameters()

    @staticmethod
    def _reshape(param: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return param.view(1, param.shape[0], *([1] * (ref.ndim - 2)))

    def reset_parameters(self):
        with torch.no_grad():
            for param in (self.alpha, self.beta, self.gamma, self.delta):
                param.zero_()

    def forward(self, x: torch.Tensor):
        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {tuple(x.shape)}"
            )
        if int(x.shape[1]) != self.channels:
            raise ValueError(
                f"GALU was initialized with C={self.channels} but received input with C={int(x.shape[1])}."
            )

        x_norm = x / self.max_input
        zero = torch.zeros(1, dtype=x.dtype, device=x.device)
        alpha = self._reshape(self.alpha, x_norm)
        beta = self._reshape(self.beta, x_norm)
        gamma = self._reshape(self.gamma, x_norm)
        delta = self._reshape(self.delta, x_norm)
        part_prelu = F.relu(x_norm) + alpha * torch.minimum(x_norm, zero)
        part_beta = beta * (
            F.relu(1.0 - torch.abs(x_norm - 1.0))
            + torch.minimum(torch.abs(x_norm - 3.0) - 1.0, zero)
        )
        part_gamma = gamma * (
            F.relu(0.5 - torch.abs(x_norm - 0.5))
            + torch.minimum(torch.abs(x_norm - 1.5) - 0.5, zero)
        )
        part_delta = delta * (
            F.relu(0.5 - torch.abs(x_norm - 2.5))
            + torch.minimum(torch.abs(x_norm - 3.5) - 0.5, zero)
        )
        z = part_prelu + part_beta + part_gamma + part_delta
        return z * self.max_input
