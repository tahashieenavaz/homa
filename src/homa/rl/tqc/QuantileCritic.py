from .module import QuantileCriticModule


class QuantileCritic:
    def __init__(self):
        self.network = QuantileCriticModule()
