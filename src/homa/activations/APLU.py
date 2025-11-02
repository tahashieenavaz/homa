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

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        broadcast_shape = [1] * x.dim()
        broadcast_shape[1] = -1  # channel dimension

        alpha = self.alpha.view(broadcast_shape)
        beta = self.beta.view(broadcast_shape)
        gamma = self.gamma.view(broadcast_shape)
        xi = self.xi.view(broadcast_shape)
        psi = self.psi.view(broadcast_shape)
        mu = self.mu.view(broadcast_shape)

        x_activated = self.relu(x)

        hinge1 = alpha * self.relu(-x + xi)
        hinge2 = beta * self.relu(-x + psi)
        hinge3 = gamma * self.relu(-x + mu)

        return x_activated + hinge1 + hinge2 + hinge3
