import torch


class HasLogits:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def logits_(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
