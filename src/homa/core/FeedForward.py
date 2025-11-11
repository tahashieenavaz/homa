import torch
from .concerns import MovesModulesToDevice
from .modules import EncoderModule


class FeedForward(MovesModulesToDevice):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        hidden_dimension: int,
        input_dimension: int,
    ):
        self.network = EncoderModule(
            input_dimension=input_dimension, hidden_dimension=hidden_dimension
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.move_modules()
