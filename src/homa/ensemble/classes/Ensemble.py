import torch
from typing import List
from ...models.wrappers import ModelWrapper


class Ensemble:
    def __init__(self, num_classes: int):
        super().__init__()
        self.models: List[torch.nn.Module] = []
        self.num_classes = num_classes

    @property
    def size(self):
        return len(self.models)

    @property
    def length(self):
        return len(self.models)

    def record(self, model: torch.nn.Module | ModelWrapper):
        self.models.append(model)

    def push(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = torch.zeros((batch_size, self.num_classes))
        for model in self.models:
            logits += model(x)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits(x)
        return torch.nn.functional.softmax(logits, dim=1)
