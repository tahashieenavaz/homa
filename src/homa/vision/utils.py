import torch
import random


def replace_modules(module, needle: torch.nn.Module, candidates: list):
    for name, child in module.named_children():
        if isinstance(child, needle):
            new_activation = random.choice(candidates)
            setattr(module, name, new_activation())
        else:
            replace_modules(child, needle, candidates)
    return module
