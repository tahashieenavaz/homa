import torch
from sklearn.metrics import f1_score as f1
from sklearn.metrics import cohen_kappa_score as kappa


class ResnetWrapper:
    def __init__(self, architecture: torch.nn.Module, lr: float, num_classes: int):
        super().__init__()
        self.model = architecture()
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

    def f1(self, x: torch.Tensor, y: torch.Tensor):
        self.model.eval()
        predictions = self.model(x)
        return f1(y, predictions)

    def kappa(self, x: torch.Tensor, y: torch.Tensor):
        self.model.eval()
        predictions = self.model(x)
        return kappa(y, predictions)
