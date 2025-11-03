from torch import Tensor, no_grad
from torch.utils.data.dataloader import DataLoader


class ReportsAccuracy:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def accuracy_tensors(self, x: Tensor, y: Tensor) -> float:
        predictions = self.predict(x)
        return (predictions == y).float().mean().item()

    def accuracy_dataloader(self, dataloader: DataLoader):
        correct, total = 0, 0
        for x, y in dataloader:
            predictions = self.predict(x)
            correct += (predictions == y).sum().item()
            total += y.numel()
        return correct / total if total > 0 else 0.0

    @no_grad()
    def accuracy(self, x: Tensor | DataLoader, y: Tensor | None = None) -> float:
        self.network.eval()
        if isinstance(x, DataLoader):
            return self.accuracy_dataloader(x)
        return self.accuracy_tensors(x, y)
