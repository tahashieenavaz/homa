import torch
import random
from ..activations import APLU, GALU, SmallGALU, MELU, WideMELU, PDELU, SReLU


class StochasticClassifier:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = [
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

    def replace_activations(self, needle: torch.Tensor) -> None:
        replacement = random.choice(self.pool)
        for parent in self.network.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, needle):
                    setattr(parent, name, replacement())
