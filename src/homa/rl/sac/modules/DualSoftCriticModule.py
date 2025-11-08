import torch
from .SoftCriticModule import SoftCriticModule


class DualSoftCriticModule(torch.nn.Module):
    def __init__(
        self, state_dimension: int, hidden_dimension: int, action_dimension: int
    ):
        super().__init__()
        self.alpha = SoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )
        self.beta = SoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.alpha(state, action), self.beta(state, action)
