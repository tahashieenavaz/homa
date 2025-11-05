import torch
from typing import Sequence, Tuple


class MexicanReLU(torch.nn.Module):
    def __init__(
        self,
        alphas_lambdas: Sequence[Tuple[float, float]],
        channels: int | None = None,
        max_input: float = 1.0,
    ):
        super().__init__()
        self.M = float(max_input)
        self.register_buffer(
            "alphas", torch.tensor([a for a, _ in alphas_lambdas], dtype=torch.float32)
        )
        self.register_buffer(
            "lambdas", torch.tensor([l for _, l in alphas_lambdas], dtype=torch.float32)
        )
        self.K = len(alphas_lambdas)

        if channels is None:
            self.register_parameter("c0", None)  # PReLU negative slope (per-channel)
            self.register_parameter("c", None)  # (C, K) coefficients
        else:
            self._init_params(channels, device=None, dtype=None)

    def _init_params(self, C: int, device, dtype):
        self.c0 = torch.nn.Parameter(torch.zeros(C, device=device, dtype=dtype))
        self.c = torch.nn.Parameter(torch.zeros(C, self.K, device=device, dtype=dtype))

    def _expand_param(p: torch.Tensor, x: torch.Tensor, n_extra: int = 0):
        shape = (
            (1, x.shape[1]) + (1,) * (x.dim() - 2) + ((p.shape[-1],) if n_extra else ())
        )
        return p.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c0 is None or self.c is None:
            self._init_params(x.shape[1], x.device, x.dtype)
        c0 = self._expand_param(self.c0, x)
        y = F.relu(x) - c0 * F.relu(-x)
        xE = x.unsqueeze(-1)
        cE = self._expand_param(self.c, x, n_extra=1)
        aE = self.alphas.to(x.device, x.dtype).view(*((1,) * x.dim()), -1)  # (..., K)
        lE = self.lambdas.to(x.device, x.dtype).view(*((1,) * x.dim()), -1)  # (..., K)
        hats = (lE * self.M - (xE - aE * self.M).abs()).clamp_min(0.0)
        y = y + (cE * hats).sum(dim=-1)
        return y
