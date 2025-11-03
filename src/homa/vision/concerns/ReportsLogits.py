import torch


class ReportsLogits:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
