import torch
from torch import nn
import torch.nn.functional as F


class PDELU(nn.Module):
    def __init__(self, channels: int, theta: float = 0.5):
        super().__init__()
        if theta == 1.0:
            raise ValueError("theta cannot be 1.0, as it would cause a division by zero.")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.channels = int(channels)
        self.theta = float(theta)
        self._power_val = 1.0 / (1.0 - self.theta)
        self.alpha = nn.Parameter(torch.empty(self.channels))
        self.reset_parameters()

    @staticmethod
    def _reshape(param: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return param.view(1, param.shape[0], *([1] * (ref.ndim - 2)))

    def reset_parameters(self):
        with torch.no_grad():
            self.alpha.fill_(0.1)

    def forward(self, x: torch.Tensor):
        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {tuple(x.shape)}"
            )
        if int(x.shape[1]) != self.channels:
            raise ValueError(
                f"PDELU was initialized with C={self.channels} but received input with C={int(x.shape[1])}."
            )

        positive_part = F.relu(x)
        inner_term = F.relu(1.0 + (1.0 - self.theta) * x)
        powered_term = torch.pow(inner_term, self._power_val)
        subtracted_term = powered_term - 1.0
        zero = torch.zeros(1, dtype=x.dtype, device=x.device)
        alpha = self._reshape(self.alpha, x)
        negative_part = alpha * torch.minimum(subtracted_term, zero)
        return positive_part + negative_part
