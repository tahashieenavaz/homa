import torch


class APLU(torch.nn.Module):
    def __init__(self, channels: int, max_init: float = 1.0):
        super(APLU, self).__init__()
        self.channels = channels

        self.alpha = torch.nn.Parameter(torch.zeros(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))
        self.gamma = torch.nn.Parameter(torch.zeros(channels))

        self.xi = torch.nn.Parameter(max_init * torch.rand(channels))
        self.psi = torch.nn.Parameter(max_init * torch.rand(channels))
        self.mu = torch.nn.Parameter(max_init * torch.rand(channels))

    def forward(self, x):
        pass
