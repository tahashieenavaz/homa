import torch
from torch import nn
import torch.nn.functional as F


class MELU(nn.Module):
    def __init__(self, maxInput: float = 1.0):
        super().__init__()
        self.maxInput = float(maxInput)
        self._num_channels = None
        self.register_parameter("alpha", None)
        self.register_parameter("beta", None)
        self.register_parameter("gamma", None)
        self.register_parameter("delta", None)
        self.register_parameter("xi", None)
        self.register_parameter("psi", None)

    def _ensure_parameters(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got {x.dim()}D with shape {tuple(x.shape)}"
            )
        c = int(x.shape[1])
        if self._num_channels is None:
            self._num_channels = c
        elif c != self._num_channels:
            raise RuntimeError(
                f"MELU was initialized with C={self._num_channels} but got C={c}. "
                "Create a new MELU for a different channel size."
            )

        if self.alpha is None:
            shape = (1, c, 1, 1)
            device, dtype = x.device, x.dtype
            for name in ("alpha", "beta", "gamma", "delta", "xi", "psi"):
                setattr(
                    self,
                    name,
                    nn.Parameter(torch.zeros(shape, dtype=dtype, device=device)),
                )

    def reset_parameters(self):
        for p in (self.alpha, self.beta, self.gamma, self.delta, self.xi, self.psi):
            if p is not None:
                with torch.no_grad():
                    p.zero_()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self._ensure_parameters(X)

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
