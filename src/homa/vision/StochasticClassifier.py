import random
import torch
from ..device import get_device
from ..activations import ERF


class StochasticClassifier:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activation_pool = [
            torch.nn.ELU,
            torch.nn.PReLU,
            torch.nn.ReLU,
            torch.nn.ReLU6,
            torch.nn.RReLU,
            torch.nn.SELU,
            torch.nn.CELU,
            torch.nn.GELU,
            torch.nn.SiLU,
            torch.nn.Mish,
        ]

    def replace_activations(self, needle: torch.nn.Module) -> None:
        for parent in self.network.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, needle):
                    replacement = random.choice(self._activation_pool)
                    if replacement in self._requires_channels:
                        channels = infer_activation_channels(parent, name, child)
                        setattr(
                            parent,
                            name,
                            replacement(channels=channels).to(get_device()),
                        )
                    else:
                        setattr(parent, name, replacement().to(get_device()))
