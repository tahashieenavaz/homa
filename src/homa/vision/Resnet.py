import torch
from .modules import ResnetModule
from .Model import Model
from .concerns import Trainable


class Resnet(Model, Trainable):
    def __init__(self, num_classes: int, lr: float):
        super().__init__()
        self.network = ResnetModule(num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr, momentum=0.9)
