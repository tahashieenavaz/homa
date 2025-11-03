import torch


class HasProbabilities:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def probabilities(self, x: torch.Tensor):
        return torch.softmax(self.logits(x), dim=1)
