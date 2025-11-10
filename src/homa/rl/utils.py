import torch


@torch.no_grad()
def soft_update(network: torch.nn.Module, target: torch.nn.Module, tau: float):
    for n, t in zip(network.parameters(), target.parameters()):
        t.data.copy_(tau * n.data + (1 - tau) * t.data)
