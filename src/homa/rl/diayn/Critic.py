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
        gamma: float,
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
        self.gamma: float = gamma

    def values(self, states: torch.Tensor, skills: torch.Tensor):
        return self.network(states, skills)

    @torch.no_grad()
    def values_(self, *args, **kwargs):
        return self.values(*args, **kwargs)

    def advantages(
        self,
        states: torch.Tensor,
        skills: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        next_states: torch.Tensor,
    ):
        values = self.values(states=states, skills=skills)
        termination_mask = 1 - terminations
        update = (
            self.gamma
            * self.values_(states=next_states, skills=skills)
            * termination_mask
        )
        return rewards + update - values

    def loss(self):
        return self.advantages().pow(2).mean()

    def train(self):
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()
