from torch import Tensor
from torch.utils.data.dataloader import DataLoader


class ReportsAccuracy:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def accuracy(self, x: Tensor | DataLoader, y: Tensor | None = None):
        pass
