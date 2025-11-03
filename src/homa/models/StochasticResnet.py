from .modules import StochasticResnetModule
from .Resnet import Resnet


class StochasticResnet(Resnet):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = StochasticResnetModule(num_classes)
