import torch
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin
import torch.nn.functional as F


class SmallGALU(LazyModuleMixin, nn.Module):
    def __init__(self, max_input: float = 1.0):
        super().__init__()
        if max_input <= 0:
            raise ValueError("max_input must be positive.")
        self.max_input = float(max_input)
        self.alpha: torch.Tensor = UninitializedParameter()
        self.beta: torch.Tensor = UninitializedParameter()
        self._num_channels = None

    def _infer_parameters(self, x: torch.Tensor):
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
                self.alpha.new_empty(
                    param_shape, dtype=x.dtype, device=x.device
                ).zero_()
            )
            self.beta = Parameter(
                self.beta.new_empty(param_shape, dtype=x.dtype, device=x.device).zero_()
            )

        self._lazy_materialized = True

    def reset_parameters(self):
        if not isinstance(self.alpha, UninitializedParameter):
            with torch.no_grad():
                self.alpha.zero_()
                self.beta.zero_()

    def forward(self, x: torch.Tensor):
        if isinstance(self.alpha, UninitializedParameter):
            self._infer_parameters(x)

        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {tuple(x.shape)}"
            )
        if self._num_channels is not None and x.shape[1] != self._num_channels:
            raise RuntimeError(
                f"SmallGALU was initialized with C={self._num_channels} but got C={x.shape[1]}. "
                "Create a new SmallGALU for a different channel size."
            )

        x_norm = x / self.max_input
        zero = torch.zeros(1, dtype=x.dtype, device=x.device)
        part_prelu = F.relu(x_norm) + self.alpha * torch.minimum(x_norm, zero)
        part_beta = self.beta * (
            F.relu(1.0 - torch.abs(x_norm - 1.0))
            + torch.minimum(torch.abs(x_norm - 3.0) - 1.0, zero)
        )
        z = part_prelu + part_beta
        return z * self.max_input
