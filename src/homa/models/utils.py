import torch
import random


def replace_relu(model: torch.nn.Module, pool: list) -> int:
    replaced = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.ReLU):
                activation_module = random.choice(pool)
                activation_instance = activation_module()
                setattr(parent, name, activation_instance)
                replaced += 1
    return replaced
