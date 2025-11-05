import torch
from ..activations import (
    SGELU,
    LaLU,
    CaLU,
    TripleStateSwish,
    GeneralizedSwish,
    ExponentialSwish,
)


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
            SGELU,
            LaLU,
            CaLU,
            TripleStateSwish,
        ]
