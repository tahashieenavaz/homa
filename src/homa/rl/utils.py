import torch


@torch.no_grad()
def soft_update(self, network: torch.nn.Module, target: torch.nn.Module):
    for s, t in zip(network.parameters(), target.parameters()):
        t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)
