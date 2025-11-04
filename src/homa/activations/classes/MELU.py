import torch
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin
import torch.nn.functional as F


class MELU(LazyModuleMixin, nn.Module):
    def __init__(self, maxInput: float = 1.0):
        super().__init__()
        self.maxInput = float(maxInput)
        self.alpha: torch.Tensor = UninitializedParameter()
        self.beta: torch.Tensor = UninitializedParameter()
        self.gamma: torch.Tensor = UninitializedParameter()
        self.delta: torch.Tensor = UninitializedParameter()
        self.xi: torch.Tensor = UninitializedParameter()
        self.psi: torch.Tensor = UninitializedParameter()
        self._num_channels = None

    def _materialize(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got {x.dim()}D with shape {tuple(x.shape)}"
            )
        c = int(x.shape[1])
        self._num_channels = c
        shape = (1, c, 1, 1)
        with torch.no_grad():
            self.alpha = Parameter(
                self.alpha.new_empty(shape, dtype=x.dtype, device=x.device).zero_()
            )
            self.beta = Parameter(
                self.beta.new_empty(shape, dtype=x.dtype, device=x.device).zero_()
            )
            self.gamma = Parameter(
                self.gamma.new_empty(shape, dtype=x.dtype, device=x.device).zero_()
            )
            self.delta = Parameter(
                self.delta.new_empty(shape, dtype=x.dtype, device=x.device).zero_()
            )
            self.xi = Parameter(
                self.xi.new_empty(shape, dtype=x.dtype, device=x.device).zero_()
            )
            self.psi = Parameter(
                self.psi.new_empty(shape, dtype=x.dtype, device=x.device).zero_()
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
        raise RuntimeError("MELU._infer_parameters: could not locate input tensor.")

    def reset_parameters(self):
        for p in (self.alpha, self.beta, self.gamma, self.delta, self.xi, self.psi):
            if not isinstance(p, UninitializedParameter):
                with torch.no_grad():
                    p.zero_()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(self.alpha, UninitializedParameter):
            self._infer_parameters(X)
        if X.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got {X.dim()}D with shape {tuple(X.shape)}"
            )
        if self._num_channels is not None and X.shape[1] != self._num_channels:
            raise RuntimeError(
                f"MELU was initialized with C={self._num_channels} but got C={X.shape[1]}. "
                "Create a new MELU for a different channel size."
            )

        X_norm = X / self.maxInput
        Y = torch.roll(X_norm, shifts=-1, dims=1)

        term1 = F.relu(X_norm)
        term2 = self.alpha * torch.clamp(X_norm, max=0)

        dist_sq_beta = (X_norm - 2) ** 2 + (Y - 2) ** 2
        dist_sq_gamma = (X_norm - 1) ** 2 + (Y - 1) ** 2
        dist_sq_delta = (X_norm - 1) ** 2 + (Y - 3) ** 2
        dist_sq_xi = (X_norm - 3) ** 2 + (Y - 1) ** 2
        dist_sq_psi = (X_norm - 3) ** 2 + (Y - 3) ** 2

        term3 = self.beta * torch.sqrt(F.relu(2 - dist_sq_beta))
        term4 = self.gamma * torch.sqrt(F.relu(1 - dist_sq_gamma))
        term5 = self.delta * torch.sqrt(F.relu(1 - dist_sq_delta))
        term6 = self.xi * torch.sqrt(F.relu(1 - dist_sq_xi))
        term7 = self.psi * torch.sqrt(F.relu(1 - dist_sq_psi))

        Z_norm = term1 + term2 + term3 + term4 + term5 + term6 + term7
        return Z_norm * self.maxInput
