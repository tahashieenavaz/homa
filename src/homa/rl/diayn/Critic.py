import torch
from .modules import CriticModule
from ...common.concerns import MovesModulesToDevice


class Critic(MovesModulesToDevice):
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        num_skills: int,
        lr: float,
        decay: float,
        gamma: float,
    ):
        self.gamma: float = gamma

        self.network = CriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = torch.nn.SmoothL1Loss()
        self.move_modules()

    def train(self, advantages: torch.Tensor):
        self.optimizer.zero_grad()
        loss = self.loss(advantages=advantages)
        loss.backward()
        self.optimizer.step()

    def loss(self, advantages: torch.Tensor):
        return advantages.pow(2).mean()

    def values(self, states: torch.Tensor, skills: torch.Tensor):
        return self.network(states, skills)

    @torch.no_grad()
    def values_(self, *args, **kwargs):
        return self.values(*args, **kwargs)
