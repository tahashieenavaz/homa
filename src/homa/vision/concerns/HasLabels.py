import torch


class HasLabels:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, x: torch.Tensor):
        return torch.argmax(self.logits(x), dim=1)

    @torch.no_grad()
    def predict_(self, x: torch.Tensor):
        return torch.argmax(self.logits(x), dim=1)
