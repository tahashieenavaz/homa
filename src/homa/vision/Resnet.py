import torch
from .modules import ResnetModule
from .ClassificationModel import ClassificationModel
from .concerns import Trainable, ReportsMetrics
from ..device import get_device


class Resnet(ClassificationModel, Trainable, ReportsMetrics):
    def __init__(self, num_classes: int, lr: float):
        super().__init__()
        self.network = ResnetModule(num_classes).to(get_device())
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr, momentum=0.9)
