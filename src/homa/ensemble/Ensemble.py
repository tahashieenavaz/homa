import torch
from typing import List
from copy import deepcopy
from collections import OrderedDict
from ..models.wrappers import ModelWrapper


class Ensemble:
    def __init__(self):
        super().__init__()
        self.state_dicts: List[OrderedDict] = []
        self.model = None

    @property
    def size(self):
        return len(self.models)

    @property
    def length(self):
        return len(self.models)

    def record(self, wrapper: ModelWrapper):
        if self.model is None:
            self.model = deepcopy(wrapper.model)
        self.state_dicts.append(wrapper.model.state_dict())

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
