import torch
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F


class APLU(nn.Module):
    def __init__(self, max_input: float = 1.0):
        super().__init__()
        self.max_input = float(max_input)
        self.alpha = UninitializedParameter()
        self.beta = UninitializedParameter()
        self.gamma = UninitializedParameter()
        self.xi = UninitializedParameter()
        self.psi = UninitializedParameter()
        self.mu = UninitializedParameter()
        self._num_channels = None

    def _initialize_parameters(self, x: torch.Tensor):
        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {tuple(x.shape)}"
            )

        channels = int(x.shape[1])
        self._num_channels = channels
        param_shape = [1] * x.ndim
        param_shape[1] = channels

        with torch.no_grad():
            self.alpha = Parameter(
                torch.zeros(param_shape, dtype=x.dtype, device=x.device)
            )
            self.beta = Parameter(
                torch.zeros(param_shape, dtype=x.dtype, device=x.device)
            )
            self.gamma = Parameter(
                torch.zeros(param_shape, dtype=x.dtype, device=x.device)
            )
            self.xi = Parameter(
                torch.empty(param_shape, dtype=x.dtype, device=x.device).uniform_(
                    0.0, self.max_input
                )
            )
            self.psi = Parameter(
                torch.empty(param_shape, dtype=x.dtype, device=x.device).uniform_(
                    0.0, self.max_input
                )
            )
            self.mu = Parameter(
                torch.empty(param_shape, dtype=x.dtype, device=x.device).uniform_(
                    0.0, self.max_input
                )
            )

    def reset_parameters(self):
        if isinstance(self.alpha, UninitializedParameter):
            return

        with torch.no_grad():
            self.alpha.zero_()
            self.beta.zero_()
            self.gamma.zero_()
            self.xi.uniform_(0.0, self.max_input)
            self.psi.uniform_(0.0, self.max_input)
            self.mu.uniform_(0.0, self.max_input)

    def forward(self, x: torch.Tensor):
        if isinstance(self.alpha, UninitializedParameter):
            self._initialize_parameters(x)

        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {tuple(x.shape)}"
            )
        if self._num_channels is not None and x.shape[1] != self._num_channels:
            raise RuntimeError(
                f"APLU was initialized with C={self._num_channels} but got C={x.shape[1]}. "
                "Create a new APLU for a different channel size."
            )

        a = F.relu(x)
        b = self.alpha * F.relu(-x + self.xi)
        c = self.beta * F.relu(-x + self.psi)
        d = self.gamma * F.relu(-x + self.mu)
        return a + b + c + d
