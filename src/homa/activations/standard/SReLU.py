import torch


class SReLU(torch.nn.Module):
    def __init__(self, channels: int):
        super(SReLU, self).__init__()

        self.alpha = torch.nn.Parameter(torch.zeros(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.delta = torch.nn.Parameter(torch.ones(channels))

    def forward(self, x):
        broadcast_shape = [1] * x.dim()
        if x.dim() > 1:
            broadcast_shape[1] = -1

        alpha = self.alpha.view(broadcast_shape)
        beta = self.beta.view(broadcast_shape)
        gamma = self.gamma.view(broadcast_shape)
        delta = self.delta.view(broadcast_shape)

        start = beta + alpha * (x - beta)
        finish = delta + gamma * (x - delta)

        x = torch.where(x < beta, start, x)
        x = torch.where(x > delta, finish, x)

        return x
