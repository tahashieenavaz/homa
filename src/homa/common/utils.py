import torch
import random
from typing import List, Type


def replace_modules(
    module, needle: torch.nn.Module, candidates: List[Type[torch.nn.Module]]
):
    for name, child in module.named_children():
        if isinstance(child, needle):
            new_activation = random.choice(candidates)
            setattr(module, name, new_activation())
        else:
            replace_modules(child, needle, candidates)
    return module
