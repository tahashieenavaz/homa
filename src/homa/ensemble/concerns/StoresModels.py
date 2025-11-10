import torch
from typing import List, Type
from collections import OrderedDict
from ...vision import Model


class StoresModels:
    def __init__(self):
        super().__init__()
        self.factories: List[Type[torch.nn.Module]] = []
        self.weights: List[OrderedDict] = []

    def record(self, model: Model | torch.nn.Module):
        model_: torch.nn.Module | None = None
        if isinstance(model, Model):
            model_ = model.network
        elif isinstance(model, torch.nn.Module):
            model_ = model
        else:
            raise TypeError("Wrong input to ensemble record")

        self.factories.append(model_.__class__)
        self.weights.append(model_.state_dict())

    def push(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def append(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.record(*args, **kwargs)
