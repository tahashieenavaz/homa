import torch
from .modules import ContinuousActorModule
from ...core.concerns import MovesNetworkToDevice


class Actor(MovesNetworkToDevice):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        num_skills: int,
        hidden_dimension: int,
        lr: float,
        decay: float,
        epsilon: float,
    ):
        self.network = ContinuousActorModule(
            state_dimension=state_dimension,
            action_dimension=action_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
            epsilon=epsilon,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = False

    def train(self):
        pass
