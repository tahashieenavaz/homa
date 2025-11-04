import torch
import random

from .APLU import APLU
from .GALU import GALU
from .SmallGALU import SmallGALU
from .MELU import MELU
from .WideMELU import WideMELU
from .PDELU import PDELU
from .SReLU import SReLU


class StochasticActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = random.choice(
            [
                APLU,
                GALU,
                SmallGALU,
                MELU,
                WideMELU,
                PDELU,
                SReLU,
                torch.nn.ReLU,
                torch.nn.PReLU,
                torch.nn.LeakyReLU,
                torch.nn.ELU,
            ]
        )
        self.gate = self.gate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)
