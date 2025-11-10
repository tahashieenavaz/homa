import torch
from ...device import move


class MovesModulesToDevice:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def move_modules(self):
        for module in dir(self):
            if module.startswith("__") or module.endswith("__"):
                continue

            attribute = getattr(self, module)
            if isinstance(attribute, torch.nn.Module):
                move(attribute)
