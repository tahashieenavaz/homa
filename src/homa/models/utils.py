import torch


def replace_modules(
    model: torch.nn.Module, find: list | torch.Tensor, replacement: torch.nn.Module
) -> int:
    if not isinstance(find, list):
        find = [find]

    replaced = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            for needle in find:
                if isinstance(child, needle):
                    setattr(parent, name, replacement())
                    replaced += 1
    return replaced


def replace_relu(model: torch.nn.Module, replacement: torch.nn.Module):
    return replace_modules(model, torch.nn.ReLU, replacement)
