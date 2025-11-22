import torch
from .NoisyLinear import NoisyLinear


class NoisyFeedForward(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation_fn=torch.nn.GELU,
    ):
        super().__init__()
        self.mu = NoisyLinear(input_dimension, hidden_dimension)
        self.xi = NoisyLinear(hidden_dimension, output_dimension)
        self.chi = activation_fn()

    def reset_noise(self):
        self.mu.reset_noise()
        self.xi.reset_noise()

    def forward(self, x):
        x = self.mu(x)
        x = self.chi(x)
        x = self.xi(x)
        return x
