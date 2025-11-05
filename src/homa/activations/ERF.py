import torch


class ERF(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha))

    def forward(self, x: torch.Tensor):
        return x * torch.erf(self.alpha * x)
