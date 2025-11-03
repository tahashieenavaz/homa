from .modules import StochasticResnetModule
from .Resnet import Resnet


class StochasticResnet(Resnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = StochasticResnetModule(kwargs["num_classes"])
