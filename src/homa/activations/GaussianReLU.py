import torch
from typing import Sequence, Tuple


class GaussianReLU(torch.nn.Module):
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
            self.register_parameter("c0", None)  # per-channel (PReLU slope)
            self.register_parameter("c", None)  # (C, K) coefficients
        else:
            self._init_params(channels, None, None)

    def _init_params(self, C: int, device, dtype):
        self.c0 = torch.nn.Parameter(torch.zeros(C, device=device, dtype=dtype))
        self.c = torch.nn.Parameter(torch.zeros(C, self.K, device=device, dtype=dtype))

    def _expand_param(p: torch.Tensor, x: torch.Tensor, add_K: bool = False):
        shape = (
            (1, x.shape[1]) + (1,) * (x.dim() - 2) + ((p.shape[-1],) if add_K else ())
        )
        return p.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c0 is None or self.c is None:
            self._init_params(x.shape[1], x.device, x.dtype)
        c0 = self._expand_param(self.c0, x)
        y = torch.nn.functional.relu(x) - c0 * torch.nn.functional.relu(-x)
        a = self.alphas.to(x.device, x.dtype).view(*((1,) * x.dim()), -1)
        l = self.lambdas.to(x.device, x.dtype).view(*((1,) * x.dim()), -1)
        xE = x.unsqueeze(-1)
        term1 = (l * self.M - (xE - a * self.M).abs()).clamp_min(0.0)
        term2 = ((xE - a * self.M - 2 * l * self.M).abs() - l * self.M).clamp_max(0.0)
        hats = term1 + term2
        c = self._expand_param(self.c, x, add_K=True)  # (1,C,...,K)
        return y + (c * hats).sum(dim=-1)
