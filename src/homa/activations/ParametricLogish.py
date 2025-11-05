import torch


class ParametricLogish(torch.nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.beta * x)
        b = torch.log(1 + a)
        return self.alpha * x * b
