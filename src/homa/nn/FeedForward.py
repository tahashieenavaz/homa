import torch
from typing import Type


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        factor: int = 4,
        output_dimension: int | None = None,
        activation: Type[torch.nn.Module] = torch.nn.GELU,
        use_skip: bool = False,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.use_layernorm = use_layernorm

        if not output_dimension:
            output_dimension = dimension

        if use_layernorm:
            self.chi = torch.nn.LayerNorm(dimension)

        self.mu = torch.nn.Linear(dimension, dimension * factor)
        self.xi = torch.nn.Linear(dimension * factor, output_dimension)
        self.sigma = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mu(x)
        x = self.sigma(x)
        x = self.xi(x)

        if self.use_skip:
            x += residual

        if self.use_layernorm:
            x = self.chi(x)

        return x
