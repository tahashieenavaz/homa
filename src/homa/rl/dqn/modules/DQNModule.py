import torch


class DQNModule(torch.nn.Module):
    def __init__(self, action_dim: int, input_channels: int, embedding_dimension: int):
        super().__init__()
        self.phi = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
        )
        self.sigma = torch.nn.Sequential(
            torch.nn.LazyLinear(embedding_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dimension, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.phi(x)
        return self.sigma(features)
