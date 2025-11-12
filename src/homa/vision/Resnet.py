import torch
from .modules import ResnetModule
from .Classifier import Classifier
from .concerns import Trainable, ReportsMetrics
from ..common.concerns import MovesModulesToDevice


class Resnet(Classifier, Trainable, ReportsMetrics, MovesModulesToDevice):
    def __init__(self, num_classes: int, lr: float = 0.001):
        super().__init__()
        self.network = ResnetModule(num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr, momentum=0.9)
        self.move_modules()
