import torch
from typing import Type


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        dimension: int,
        factor: int | None = None,
        output_dimension: int | None = None,
        embedding_dimension: int | None = None,
        activation: Type[torch.nn.Module] = torch.nn.GELU,
        use_skip: bool = False,
        use_layernorm: bool = False,
    ):
        super().__init__()

        if factor is not None and embedding_dimension is not None:
            raise Exception(
                "You should specify either factor or embedding dimensions in homa.nn.FeedForward. Got both."
            )

        if factor is not None and embedding_dimension is None:
            sigma = factor * dimension
        else:
            sigma = embedding_dimension

        self.use_skip = use_skip
        self.use_layernorm = use_layernorm

        if not output_dimension:
            output_dimension = dimension

        if use_layernorm:
            self.chi = torch.nn.LayerNorm(dimension)

        self.mu = torch.nn.Linear(dimension, sigma)
        self.xi = torch.nn.Linear(sigma, output_dimension)
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
