import torch
from typing import Type


class EncoderModule(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        activation: Type[torch.nn.Module] = torch.nn.GELU,
    ):
        super().__init__()

        self.stream = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.LayerNorm(hidden_dimension),
            activation(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.LayerNorm(hidden_dimension),
            activation(),
        )

    def forward(self, x):
        return self.stream(x)
