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
        self.criterion = torch.nn.SmoothL1Loss()
