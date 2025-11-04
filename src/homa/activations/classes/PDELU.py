import torch
from torch import nn
import torch.nn.functional as F


class PDELU(nn.Module):
    def __init__(self, theta: float = 0.5):
        super().__init__()
        if theta == 1.0:
            raise ValueError(
                "theta cannot be 1.0, as it would cause a division by zero."
            )
        self.theta = float(theta)
        self._power_val = 1.0 / (1.0 - self.theta)
        self.register_parameter("alpha", None)
        self._num_channels = None

    def _ensure_parameters(self, x: torch.Tensor):
        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {tuple(x.shape)}"
            )

        c = int(x.shape[1])
        if self._num_channels is None:
            self._num_channels = c
        elif c != self._num_channels:
            raise RuntimeError(
                f"PDELU was initialized with C={self._num_channels} but got C={c}. "
                "Create a new PDELU for a different channel size."
            )

        if self.alpha is None:
            param_shape = [1] * x.ndim
            param_shape[1] = c
            self.alpha = nn.Parameter(
                torch.full(param_shape, 0.1, dtype=x.dtype, device=x.device)
            )

    def reset_parameters(self):
        if self.alpha is not None:
            with torch.no_grad():
                self.alpha.fill_(0.1)

    def forward(self, x: torch.Tensor):
        self._ensure_parameters(x)

        positive_part = F.relu(x)
        inner_term = F.relu(1.0 + (1.0 - self.theta) * x)
        powered_term = torch.pow(inner_term, self._power_val)
        subtracted_term = powered_term - 1.0
        zero = torch.zeros(1, dtype=x.dtype, device=x.device)
        negative_part = self.alpha * torch.minimum(subtracted_term, zero)
        return positive_part + negative_part
