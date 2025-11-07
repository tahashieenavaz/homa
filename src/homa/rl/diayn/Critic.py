import torch
from .modules import CriticModule
from ...core.concerns import MovesNetworkToDevice


class Critic(MovesNetworkToDevice):
    def __init__(
        self,
        state_dimension: int,
        num_skills: int,
        hidden_dimension: int,
        lr: float,
        decay: float,
    ):
        self.network = CriticModule(
            state_dimension=state_dimension,
            num_skills=num_skills,
            hidden_dimension=hidden_dimension,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = torch.nn.SmoothL1Loss()

    def forward(self, state: torch.Tensor, skill: torch.Tensor):
        pass
