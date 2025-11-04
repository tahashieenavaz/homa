import random
import torch
from ..activations import (
    APLU,
    GALU,
    SmallGALU,
    MELU,
    WideMELU,
    PDELU,
    SReLU,
    infer_activation_channels,
)


class StochasticClassifier:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activation_pool = [
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
        self._requires_channels = {
            APLU,
            GALU,
            SmallGALU,
            MELU,
            WideMELU,
            PDELU,
            SReLU,
        }

    def replace_activations(self, needle: torch.nn.Module) -> None:
        for parent in self.network.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, needle):
                    replacement = random.choice(self._activation_pool)
                    if replacement in self._requires_channels:
                        channels = infer_activation_channels(parent, name, child)
                        setattr(parent, name, replacement(channels=channels))
                    else:
                        setattr(parent, name, replacement())
