import torch
from typing import Type


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        output_dimension: int,
        activation: Type[torch.nn.Module],
    ):
        super().__init__()

        self.mu = torch.nn.LazyLinear(embedding_dimension)
        self.xi = torch.nn.LazyLinear(output_dimension)
        self.sigma = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mu(x)
        x = self.sigma(x)
        x = self.xi(x)
        return x
