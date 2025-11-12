import torch
from ...device import get_device


class MovesModulesToDevice:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def move_modules(self):
        for module in dir(self):
            if module.startswith("__") or module.endswith("__"):
                continue

            attribute = getattr(self, module)
            _device = get_device()
            if (
                isinstance(attribute, torch.nn.Module)
                or isinstance(attribute, torch.nn.Parameter)
                or isinstance(attribute, torch.Tensor)
            ):
                if hasattr(attribute, "to"):
                    try:
                        setattr(self, module, attribute.to(_device))
                    except (AttributeError, TypeError):
                        # it passes properties
                        pass
