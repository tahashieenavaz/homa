import torch


class Predicts:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, x: torch.Tensor):
        return torch.softmax(self.logits(x), dim=1)
