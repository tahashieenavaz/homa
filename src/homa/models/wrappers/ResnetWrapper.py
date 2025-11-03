import torch
from .ModelWrapper import ModelWrapper


class ResnetWrapper(ModelWrapper):
    def __init__(self, architecture: torch.nn.Module, lr: float, num_classes: int):
        super().__init__()
        self.architecture = architecture
        self.model = self.architecture()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def train(self, x: torch.Tensor, y: torch.Tensor):
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.model(x)
        loss = self.criterion(predictions, y)
        loss.backward()
        self.optimizer.step()
