import torch
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F


class GALU(nn.Module):
    def __init__(self, max_input: float = 1.0):
        super().__init__()
        if max_input <= 0:
            raise ValueError("max_input must be positive.")
        self.max_input = float(max_input)
        self.alpha: torch.Tensor = UninitializedParameter()
        self.beta: torch.Tensor = UninitializedParameter()
        self.gamma: torch.Tensor = UninitializedParameter()
        self.delta: torch.Tensor = UninitializedParameter()

    def _initialize_parameters(self, x: torch.Tensor):
        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {tuple(x.shape)}"
            )
        param_shape = [1] * x.ndim
        param_shape[1] = int(x.shape[1])
        zeros = torch.zeros(param_shape, dtype=x.dtype, device=x.device)
        with torch.no_grad():
            for name in ("alpha", "beta", "gamma", "delta"):
                setattr(self, name, Parameter(zeros.clone()))

    def reset_parameters(self):
        for name in ("alpha", "beta", "gamma", "delta"):
            p = getattr(self, name)
            if not isinstance(p, UninitializedParameter):
                with torch.no_grad():
                    p.zero_()

    def forward(self, x: torch.Tensor):
        if isinstance(self.alpha, UninitializedParameter):
            self._initialize_parameters(x)

        if x.ndim < 2:
            raise ValueError(
                f"Input tensor must have at least 2 dimensions (N, C), but got shape {tuple(x.shape)}"
            )
        if not isinstance(self.alpha, UninitializedParameter) and x.shape[1] != self.alpha.shape[1]:
            raise RuntimeError(
                f"GALU was initialized with C={self.alpha.shape[1]} but got C={x.shape[1]}. "
                "Create a new GALU for a different channel size."
            )

        x_norm = x / self.max_input
        zero = x.new_zeros(1)
        part_prelu = F.relu(x_norm) + self.alpha * torch.minimum(x_norm, zero)
        part_beta = self.beta * (
            F.relu(1.0 - torch.abs(x_norm - 1.0))
            + torch.minimum(torch.abs(x_norm - 3.0) - 1.0, zero)
        )
        part_gamma = self.gamma * (
            F.relu(0.5 - torch.abs(x_norm - 0.5))
            + torch.minimum(torch.abs(x_norm - 1.5) - 0.5, zero)
        )
        part_delta = self.delta * (
            F.relu(0.5 - torch.abs(x_norm - 2.5))
            + torch.minimum(torch.abs(x_norm - 3.5) - 0.5, zero)
        )
        z = part_prelu + part_beta + part_gamma + part_delta
        return z * self.max_input
