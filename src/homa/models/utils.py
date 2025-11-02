import torch


def replace_relu(model: torch.nn.Module, activation: torch.nn.Module) -> int:
    replaced = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.ReLU):
                setattr(parent, name, activation())
                replaced += 1
    return replaced
