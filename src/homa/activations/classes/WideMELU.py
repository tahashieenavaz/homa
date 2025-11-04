import torch
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin
import torch.nn.functional as F


class WideMELU(LazyModuleMixin, nn.Module):
    def __init__(self, maxInput: float = 1.0):
        super().__init__()
        self.maxInput = float(maxInput)
        self.alpha: torch.Tensor = UninitializedParameter()
        self.beta: torch.Tensor = UninitializedParameter()
        self.gamma: torch.Tensor = UninitializedParameter()
        self.delta: torch.Tensor = UninitializedParameter()
        self.xi: torch.Tensor = UninitializedParameter()
        self.psi: torch.Tensor = UninitializedParameter()
        self.theta: torch.Tensor = UninitializedParameter()
        self.lam: torch.Tensor = UninitializedParameter()
        self.num_channels = None

    def _infer_parameters(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got {x.dim()}D with shape {tuple(x.shape)}"
            )

        c = int(x.shape[1])
        self.num_channels = c
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
            self.theta = Parameter(
                self.theta.new_empty(shape, dtype=x.dtype, device=x.device).zero_()
            )
            self.lam = Parameter(
                self.lam.new_empty(shape, dtype=x.dtype, device=x.device).zero_()
            )

        self._lazy_materialized = True

    def reset_parameters(self):
        params = (
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            self.xi,
            self.psi,
            self.theta,
            self.lam,
        )
        for p in params:
            if not isinstance(p, UninitializedParameter):
                with torch.no_grad():
                    p.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.alpha, UninitializedParameter):
            self._infer_parameters(x)

        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got {x.dim()}D with shape {tuple(x.shape)}"
            )
        if self.num_channels is not None and x.shape[1] != self.num_channels:
            raise RuntimeError(
                f"WideMELU was initialized with C={self.num_channels} but got C={x.shape[1]}."
                " Create a new WideMELU for different channel sizes."
            )

        X_norm = x / self.maxInput
        Y = torch.roll(X_norm, shifts=-1, dims=1)

        term1 = F.relu(X_norm)
        term2 = self.alpha * torch.clamp(X_norm, max=0)
        dist_sq_beta = (X_norm - 2) ** 2 + (Y - 2) ** 2
        dist_sq_gamma = (X_norm - 1) ** 2 + (Y - 1) ** 2
        dist_sq_delta = (X_norm - 1) ** 2 + (Y - 3) ** 2
        dist_sq_xi = (X_norm - 3) ** 2 + (Y - 1) ** 2
        dist_sq_psi = (X_norm - 3) ** 2 + (Y - 3) ** 2
        dist_sq_theta = (X_norm - 1) ** 2 + (Y - 2) ** 2
        dist_sq_lambda = (X_norm - 3) ** 2 + (Y - 2) ** 2
        term3 = self.beta * torch.sqrt(F.relu(2 - dist_sq_beta))
        term4 = self.gamma * torch.sqrt(F.relu(1 - dist_sq_gamma))
        term5 = self.delta * torch.sqrt(F.relu(1 - dist_sq_delta))
        term6 = self.xi * torch.sqrt(F.relu(1 - dist_sq_xi))
        term7 = self.psi * torch.sqrt(F.relu(1 - dist_sq_psi))
        term8 = self.theta * torch.sqrt(F.relu(1 - dist_sq_theta))
        term9 = self.lam * torch.sqrt(F.relu(1 - dist_sq_lambda))
        Z_norm = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
        return Z_norm * self.maxInput
