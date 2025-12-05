import torch
from .ActivationFunction import ActivationFunction


class AdaptiveActivationFunction(ActivationFunction):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        arguments_text = ""

        if hasattr(self, "num_channels"):
            arguments_text = f"channels={self.num_channels}"

        return f"{__class__.__name__}({arguments_text})"
