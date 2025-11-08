import torch
from .modules import DualSoftCriticModule
from typing import Type


class SoftCritic:
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        lr: float,
        weight_decay: float,
        tau: float,
    ):
        self.tau = tau

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

    def soft_update(
        self, network: Type[torch.nn.Module], target: Type[torch.nn.Module]
    ):
        for s, t in zip(network.parameters(), target.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def update(self):
        self.soft_update(self.network.alpha, self.target.alpha)
        self.soft_update(self.network.beta, self.target.beta)
