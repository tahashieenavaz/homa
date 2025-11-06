import torch
import io
from typing import List
from ...vision import Model


class StoresModels:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models: List[torch.nn.Module] = []

    def record(self, model: Model | torch.nn.Module):
        model_: torch.nn.Module | None = None
        if isinstance(model, Model):
            model_ = model.network
        elif isinstance(model, torch.nn.Module):
            model_ = model
        else:
            raise TypeError("Wrong input to ensemble record")

        device = model_.device
        buffer = io.BytesIO()
        torch.save(model_.to("cpu"), buffer)
        buffer.seek(0)
        model_ = torch.load(buffer, map_location=device)
        self.models.append(model_)

    def push(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def append(self, *args, **kwargs):
        self.record(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.record(*args, **kwargs)
