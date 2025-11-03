import torch
from .modules import ResnetModule


class Resnet:
    def __init__(self, num_classes: int, lr: float):
        super().__init__()
        self.model = ResnetModule(num_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, x: torch.Tensor, y: torch.Tensor):
        self.optimizer.zero_grad()
        loss = self.criterion(x, y)
        loss.backward()
        self.optimizer.step()
