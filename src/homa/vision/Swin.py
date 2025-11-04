import torch
from .ClassificationModel import ClassificationModel
from .modules import SwinModule


class Swin(ClassificationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = SwinMoudle()
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.0001)
