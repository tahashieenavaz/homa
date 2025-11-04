import torch
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin
import torch.nn.functional as F


class PDELU(LazyModuleMixin, nn.Module):
    def __init__(self, theta: float = 0.5):
        super().__init__()
        if theta == 1.0:
            raise ValueError(
                "theta cannot be 1.0, as it would cause a division by zero."
            )
        self.theta = float(theta)
        self._power_val = 1.0 / (1.0 - self.theta)
        self.alpha: torch.Tensor = UninitializedParameter()
        self._num_channels = None

    def _infer_parameters(self, *args, **kwargs):
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            x = args[0]
        elif (
            len(args) >= 2 and isinstance(args[1], (tuple, list)) and len(args[1]) >= 1
        ):
            x = args[1][0]
        else:
            x = kwargs.get("input", None)
            if isinstance(x, (tuple, list)):
                x = x[0]

        if not isinstance(x, torch.Tensor):
            raise RuntimeError(
                "PDELU._infer_parameters: could not locate input tensor."
            )

        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {tuple(x.shape)}"
            )

        c = int(x.shape[1])
        self._num_channels = c
        param_shape = [1] * x.ndim
        param_shape[1] = c

        with torch.no_grad():
            self.alpha = Parameter(
                self.alpha.new_empty(param_shape, dtype=x.dtype, device=x.device).fill_(
                    0.1
                )
            )
        self._lazy_materialized = True

    def reset_parameters(self):
        if not isinstance(self.alpha, UninitializedParameter):
            with torch.no_grad():
                self.alpha.fill_(0.1)

    def forward(self, x: torch.Tensor):
        if isinstance(self.alpha, UninitializedParameter):
            self._infer_parameters(x)

        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {tuple(x.shape)}"
            )
        if self._num_channels is not None and x.shape[1] != self._num_channels:
            raise RuntimeError(
                f"PDELU was initialized with C={self._num_channels} but got C={x.shape[1]}. "
                "Create a new PDELU for a different channel size."
            )
        positive_part = F.relu(x)
        inner_term = F.relu(1.0 + (1.0 - self.theta) * x)
        powered_term = torch.pow(inner_term, self._power_val)
        subtracted_term = powered_term - 1.0
        zero = torch.zeros(1, dtype=x.dtype, device=x.device)
        negative_part = self.alpha * torch.minimum(subtracted_term, zero)
        return positive_part + negative_part
