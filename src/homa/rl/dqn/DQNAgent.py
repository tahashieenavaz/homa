import torch
from .modules import DQNModule


class DQNAgent:
    def __init__(
        self,
        action_dim: int,
        embedding_dimension: int,
        input_channels: int,
        lr: float,
        weight_decay: float,
    ):
        self.network = DQNModule(
            action_dim=action_dim,
            embedding_dimension=embedding_dimension,
            input_channels=input_channels,
        )
        self.optimizer = torch.nn.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = torch.nn.SmoothL1Loss()

    def train(self):
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()

    def loss(self):
        pass
