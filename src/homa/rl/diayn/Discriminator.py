import torch
from .modules import DiscriminatorModule
from ...core.concerns import MovesNetworkToDevice


class Discriminator(MovesNetworkToDevice):
    def __init__(
        self,
        decay: float,
        lr: float,
        state_dimension: int,
        hidden_dimension: int,
        num_skills: int,
    ):
        self.network = DiscriminatorModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            num_skills=num_skills,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def loss(self, states: torch.Tensor, skills_indices: torch.Tensor):
        logits = self.network(states)
        return self.criterion(logits, skills_indices)

    def train(self, states: torch.Tensor, skills_indices: torch.Tensor):
        self.optimizer.zero_grad()
        loss = self.loss(states=states, skills_indices=skills_indices)
        loss.backward()
        self.optimizer.step()
