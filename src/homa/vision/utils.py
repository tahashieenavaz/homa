import torch
import random


def replace_activations(module, needle: torch.nn.Module, candidates: list):
    for name, module in module.named_children():
        if isinstance(module, needle):
            factory = random.choice(candidates)
            new_module = factory()
            setattr(module, name, new_module)
        else:
            replace_activations(module, needle, candidates)
