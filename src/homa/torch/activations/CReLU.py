import torch


class CReLU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.cat((torch.relu(x), torch.relu(-x)), dim=1)
