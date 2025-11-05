import torch
from .Classifier import Classifier
from .concerns import Trainable, ReportsMetrics
from .modules import SwinModule
from ..device import get_device


class Swin(Classifier, Trainable, ReportsMetrics):
    def __init__(self, num_classes: int, lr: float = 0.0001):
        super().__init__()
        self.network = SwinModule(num_classes=num_classes).to(get_device())
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
