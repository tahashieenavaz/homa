from ..activations import (
    AOAF,
    AReLU,
    DPReLU,
    DualLine,
    FReLU,
    LeLeLU,
    PERU,
    PiLU,
    ShiLU,
    StarReLU,
)


class StochasticClassifier:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activation_pool = [
            AOAF,
            AReLU,
            DPReLU,
            DualLine,
            FReLU,
            LeLeLU,
            PERU,
            PiLU,
            ShiLU,
            StarReLU,
        ]
