import torch


@torch.no_grad()
def soft_update(network: torch.nn.Module, target: torch.nn.Module, tau: float):
    for s, t in zip(network.parameters(), target.parameters()):
        t.data.copy_(tau * s.data + (1 - tau) * t.data)
