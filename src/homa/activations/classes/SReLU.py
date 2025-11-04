import torch
from torch import nn


class SReLU(nn.Module):
    def __init__(
        self,
        alpha_init: float = 0.0,
        beta_init: float = 0.0,
        gamma_init: float = 1.0,
        delta_init: float = 1.0,
    ):
        super().__init__()
        self.alpha_init_val = float(alpha_init)
        self.beta_init_val = float(beta_init)
        self.gamma_init_val = float(gamma_init)
        self.delta_init_val = float(delta_init)
        self._num_channels = None
        self.register_parameter("alpha", None)
        self.register_parameter("beta", None)
        self.register_parameter("gamma", None)
        self.register_parameter("delta", None)

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
                f"SReLU was initialized with C={self._num_channels} but got C={c}. "
                "Create a new SReLU for different channel sizes."
            )

        if self.alpha is None:
            shape = (1, c, 1, 1)
            device, dtype = x.device, x.dtype
            self.alpha = nn.Parameter(
                torch.full(shape, self.alpha_init_val, dtype=dtype, device=device)
            )
            self.beta = nn.Parameter(
                torch.full(shape, self.beta_init_val, dtype=dtype, device=device)
            )
            self.gamma = nn.Parameter(
                torch.full(shape, self.gamma_init_val, dtype=dtype, device=device)
            )
            self.delta = nn.Parameter(
                torch.full(shape, self.delta_init_val, dtype=dtype, device=device)
            )

    def reset_parameters(self):
        if self.alpha is not None:
            with torch.no_grad():
                self.alpha.fill_(self.alpha_init_val)
                self.beta.fill_(self.beta_init_val)
                self.gamma.fill_(self.gamma_init_val)
                self.delta.fill_(self.delta_init_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_parameters(x)

        start = self.beta + self.alpha * (x - self.beta)
        finish = self.delta + self.gamma * (x - self.delta)
        out = torch.where(x < self.beta, start, x)
        out = torch.where(x > self.delta, finish, out)
        return out
