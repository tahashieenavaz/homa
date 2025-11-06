from ..activations import AOAF, FReLU, LeLeLU, PERU


class StochasticClassifier:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activation_pool = [AOAF, FReLU, LeLeLU, PERU]
