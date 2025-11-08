import torch
from .modules import DualSoftCriticModule


class SoftCritic:
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        lr: float,
        weight_decay: float,
    ):
        self.network = DualSoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )
        self.target = DualSoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self):
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()

    def loss(self):
        pass
