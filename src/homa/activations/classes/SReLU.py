import torch
from torch.nn.parameter import Parameter, UninitializedParameter


class SReLU(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
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
        self.alpha: torch.Tensor = UninitializedParameter()
        self.beta: torch.Tensor = UninitializedParameter()
        self.gamma: torch.Tensor = UninitializedParameter()
        self.delta: torch.Tensor = UninitializedParameter()

        # track channels once inferred
        self.num_channels = None

    def _infer_parameters(self, x: torch.Tensor):
        if x.dim() < 2:
            raise ValueError(
                f"Input must have shape (N, C, ...). Got dim={x.dim()} with shape {tuple(x.shape)}"
            )

        c = x.shape[1]
        self.num_channels = int(c)

        param_shape = [1] * x.dim()
        param_shape[1] = c

        with torch.no_grad():
            self.alpha = Parameter(
                self.alpha.new_empty(param_shape, dtype=x.dtype, device=x.device).fill_(
                    self.alpha_init_val
                )
            )
            self.beta = Parameter(
                self.beta.new_empty(param_shape, dtype=x.dtype, device=x.device).fill_(
                    self.beta_init_val
                )
            )
            self.gamma = Parameter(
                self.gamma.new_empty(param_shape, dtype=x.dtype, device=x.device).fill_(
                    self.gamma_init_val
                )
            )
            self.delta = Parameter(
                self.delta.new_empty(param_shape, dtype=x.dtype, device=x.device).fill_(
                    self.delta_init_val
                )
            )

        self._lazy_materialized = True

    def reset_parameters(self):
        if not isinstance(self.alpha, UninitializedParameter):
            with torch.no_grad():
                self.alpha.fill_(self.alpha_init_val)
                self.beta.fill_(self.beta_init_val)
                self.gamma.fill_(self.gamma_init_val)
                self.delta.fill_(self.delta_init_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.alpha, UninitializedParameter):
            self._infer_parameters(x)

        start = self.beta + self.alpha * (x - self.beta)
        finish = self.delta + self.gamma * (x - self.delta)
        out = torch.where(x < self.beta, start, x)
        out = torch.where(x > self.delta, finish, out)
        return out
