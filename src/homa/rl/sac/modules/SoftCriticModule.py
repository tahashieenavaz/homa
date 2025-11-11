import torch
from ....core.modules import FeedForwardModule


class SoftCriticModule(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
    ):
        super().__init__()

        self.embedding = FeedForwardModule(
            input_dimension=state_dimension, hidden_dimension=hidden_dimension
        )
        self.phi = torch.nn.Linear(
            action_dimension + hidden_dimension, hidden_dimension
        )
        self.fc = torch.nn.Linear(hidden_dimension, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        state = self.embedding(state)
        features = torch.cat([state, action], dim=1)
        features = self.phi(features)
        features = torch.tanh(features)
        return self.fc(features)
