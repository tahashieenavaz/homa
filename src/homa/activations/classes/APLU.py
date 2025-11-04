import torch
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin
import torch.nn.functional as F


class APLU(LazyModuleMixin, nn.Module):
    def __init__(self, max_input: float = 1.0):
        super().__init__()
        self.max_input = float(max_input)
        self.alpha: torch.Tensor = UninitializedParameter()
        self.beta: torch.Tensor = UninitializedParameter()
        self.gamma: torch.Tensor = UninitializedParameter()

        # hinge locations (learnable), initialized ~ U(0, max_input)
        self.xi: torch.Tensor = UninitializedParameter()
        self.psi: torch.Tensor = UninitializedParameter()
        self.mu: torch.Tensor = UninitializedParameter()

        self._num_channels = None

    def _materialize(self, x: torch.Tensor):
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
                torch.zeros(param_shape, dtype=x.dtype, device=x.device)
            )
            self.beta = Parameter(
                torch.zeros(param_shape, dtype=x.dtype, device=x.device)
            )
            self.gamma = Parameter(
                torch.zeros(param_shape, dtype=x.dtype, device=x.device)
            )
            # hinge positions ~ U(0, max_input)
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

        self._lazy_materialized = True

    def _infer_parameters(self, *args, **kwargs):
        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            return self._materialize(args[0])
        if len(args) >= 2 and isinstance(args[1], (tuple, list)) and len(args[1]) >= 1:
            return self._materialize(args[1][0])
        inp = kwargs.get("input", None)
        if (
            isinstance(inp, (tuple, list))
            and len(inp) >= 1
            and isinstance(inp[0], torch.Tensor)
        ):
            return self._materialize(inp[0])
        raise RuntimeError("APLU._infer_parameters: could not locate input tensor.")

    def reset_parameters(self):
        if not isinstance(self.alpha, UninitializedParameter):
            with torch.no_grad():
                self.alpha.zero_()
                self.beta.zero_()
                self.gamma.zero_()
                self.xi.uniform_(0.0, self.max_input)
                self.psi.uniform_(0.0, self.max_input)
                self.mu.uniform_(0.0, self.max_input)

    def forward(self, x: torch.Tensor):
        if isinstance(self.alpha, UninitializedParameter):
            self._infer_parameters(x)

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
