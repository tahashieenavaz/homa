import torch


class Smish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.tanh(torch.log1p(torch.sigmoid(x)))
