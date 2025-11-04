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
        self.num_channels = None

    def _materialize(self, x: torch.Tensor):
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
        raise RuntimeError("_infer_parameters: could not locate input tensor")

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
