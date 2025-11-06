import torch


class AReLU(torch.nn.Module):
    def __init__(self):
        super(AReLU, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.9))
        self.b = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, z):
        negative_slope = torch.clamp(self.a, 0.01, 0.99)
        positive_slope = 1 + torch.sigmoid(self.b)
        positive = positive_slope * torch.relu(z)
        negative = negative_slope * (-torch.relu(-z))
        return positive + negative
