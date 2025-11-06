import torch
import random


def replace_activations(module, needle: torch.nn.Module, candidates: list):
    for name, child in module.named_children():
        if isinstance(child, needle):
            new_activation = random.choice(candidates)
            setattr(module, name, new_activation())
        else:
            replace_activations(child, needle, candidates)
    return module
