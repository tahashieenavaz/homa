import torch


class Logish:
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.log1p(torch.sigmoid(x))
