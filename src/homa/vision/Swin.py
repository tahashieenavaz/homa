import torch
from .ClassificationModel import ClassificationModel
from .modules import SwinModule


class Swin(ClassificationModel):
    def __init__(self, num_classes: int, lr: float = 0.0001):
        super().__init__()
        self.network = SwinModule(num_classes=num_classes)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
