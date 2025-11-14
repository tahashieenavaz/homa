import torch
from .Classifier import Classifier
from .concerns import Trainable, ReportsMetrics
from .modules import SwinModule
from ..common.concerns import MovesModulesToDevice


class Swin(Classifier, Trainable, ReportsMetrics, MovesModulesToDevice):
    def __init__(
        self,
        num_classes: int,
        lr: float = 0.0001,
        decay: float = 0.0,
        variant: str = "base",
        weights="DEFAULT",
    ):
        super().__init__()

        self.num_classes: int = num_classes

        self.network = SwinModule(
            num_classes=self.num_classes, variant=variant, weights=weights
        )
        self.move_modules()

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()
