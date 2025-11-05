import torch
from typing import Type


def replace_layers(
    module: torch.nn.Module,
    target_class: Type[torch.nn.Module],
    replacement_class: Type[torch.nn.Module],
) -> None:
    for name, child in module.named_children():
        if isinstance(child, target_class):
            inplace = getattr(child, "inplace", False)
            try:
                new_layer = replacement_class(inplace=inplace)
            except TypeError:
                try:
                    new_layer = replacement_class()
                except:
                    continue
            setattr(module, name, new_layer)
        else:
            replace_layers(child, target_class, replacement_class)
