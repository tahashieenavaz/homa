from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from .HasLogits import HasLogits
from .HasProbabilities import HasProbabilities
from .HasLabels import HasLabels
from ...device import get_device


class Trainable(HasLogits, HasProbabilities, HasLabels):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, x: Tensor | DataLoader, y: Tensor | None = None):
        if y is None and isinstance(x, DataLoader):
            self.train_dataloader(x)
            return
        self.train_tensors(x, y)

    def train_tensors(self, x: Tensor, y: Tensor):
        self.network.train()
        self.optimizer.zero_grad()
        loss = self.criterion(self.network(x).float(), y)
        loss.backward()
        self.optimizer.step()

    def train_dataloader(self, dataloader: DataLoader):
        for x, y in dataloader:
            x, y = x.to(get_device()), y.to(get_device())
            self.train_tensors(x, y)
