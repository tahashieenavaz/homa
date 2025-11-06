import torch
from typing import List


class ChannelBased:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False
        self.num_channels = None

    def initialize(self, x: torch.Tensor, attrs: List[str] | str):
        if getattr(self, "_initialized", False):
            return
        self.num_channels = x.shape[1]

        if isinstance(attrs, str):
            attrs = [attrs]

        for attr in attrs:
            param = torch.nn.Parameter(
                torch.ones(self.num_channels, requires_grad=True)
            )
            setattr(self, attr, param)
        self._initialized = True

    def parameter_shape(self, x: torch.Tensor) -> tuple | None:
        if hasattr(self, "num_channels"):
            return (1, self.num_channels) + (1,) * (x.ndim - 2)
        return None
