import torch
from .modules import ResnetModule
from .Model import Model


class Resnet(Model):
    def __init__(self, num_classes: int, lr: float):
        super().__init__()
        self.network = ResnetModule(num_classes)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, x: torch.Tensor, y: torch.Tensor):
        self.network.train()
        self.optimizer.zero_grad()
        loss = self.criterion(x, y)
        loss.backward()
        self.optimizer.step()
