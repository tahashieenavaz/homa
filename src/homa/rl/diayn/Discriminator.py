import torch
from .modules import DiscriminatorModule
from ...common.concerns import MovesModulesToDevice


class Discriminator(MovesModulesToDevice):
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        num_skills: int,
        decay: float,
        lr: float,
    ):
        self.num_skills: int = num_skills

        self.network = DiscriminatorModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
        )
        self.move_modules()

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def loss(self, states: torch.Tensor, skills: torch.Tensor):
        logits = self.network(states)
        return self.criterion(logits, skills)

    def train(self, states: torch.Tensor, skills: torch.Tensor):
        self.optimizer.zero_grad()
        loss = self.loss(states=states, skills=skills)
        loss.backward()
        self.optimizer.step()
