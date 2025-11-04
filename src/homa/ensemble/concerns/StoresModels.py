import torch
from copy import deepcopy
from typing import List
from ...vision import Model


class StoresModels:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models: List[torch.nn.Module] = []

    def record(self, model: Model | torch.nn.Module):
        model_: torch.nn.Module | None = None
        if isinstance(model, Model):
            model_ = deepcopy(model.network)
        elif isinstance(model, torch.nn.Module):
            model_ = deepcopy(model)
        else:
            raise TypeError("Wrong input to ensemble record")
        self.models.append(model_)

    def push(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def append(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.record(*args, **kwargs)
