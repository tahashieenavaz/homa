import torch
from ....common.modules import EncoderModule


class SoftCriticModule(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
    ):
        super().__init__()

        self.phi = EncoderModule(
            input_dimension=state_dimension, hidden_dimension=hidden_dimension
        )
        self.chi = EncoderModule(
            input_dimension=action_dimension + hidden_dimension,
            hidden_dimension=hidden_dimension,
        )
        self.fc = torch.nn.Linear(hidden_dimension, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        state = self.phi(state)
        features = torch.cat([state, action], dim=1)
        features = self.chi(features)
        return self.fc(features)
