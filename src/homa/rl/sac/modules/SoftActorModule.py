import torch
from ....common.modules import EncoderModule


class SoftActorModule(torch.nn.Module):
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        min_std: float,
        max_std: float,
    ):
        super().__init__()

        self.state_dimension: int = state_dimension
        self.hidden_dimension: int = hidden_dimension
        self.action_dimension: int = action_dimension
        self.min_std: float = float(min_std)
        self.max_std: float = float(max_std)
        self.std_difference: float = self.max_std - self.min_std

        self.phi = EncoderModule(
            input_dimension=state_dimension, hidden_dimension=hidden_dimension
        )

        self.mu = torch.nn.Linear(self.hidden_dimension, self.action_dimension)
        torch.nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        torch.nn.init.constant_(self.mu.bias, 0.0)

        self.xi = torch.nn.Linear(self.hidden_dimension, self.action_dimension)
        torch.nn.init.constant_(self.xi.weight, 0.0)
        torch.nn.init.constant_(self.xi.bias, -0.5)

    def forward(self, state: torch.Tensor):
        features = self.phi(state)
        mean = self.mu(features)
        log_std = self.xi(features)
        log_std = self.min_std + (log_std + 1) * 0.5 * self.std_difference
        return mean, log_std
