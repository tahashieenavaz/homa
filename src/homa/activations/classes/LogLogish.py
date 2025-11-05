import torch


class LogLogish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        a = -torch.exp(x)
        return x * (1 - torch.exp(a))
