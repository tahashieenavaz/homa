import torch
from typing import List


class ChannelBased:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False
        self.num_channels = None

    def initialize(
        self, x: torch.Tensor, attrs: List[str] | str, values: List[float] | float = []
    ):
        if getattr(self, "_initialized", False):
            return

        if not isinstance(values, list):
            values = [values]

        if not isinstance(attrs, list):
            attrs = [attrs]

        self.num_channels = x.shape[1]
        device = x.device
        for index, attr in enumerate(attrs):
            if index < len(values) and values[index] is not None:
                default_value = float(values[index])
            else:
                default_value = 1.0
            param = torch.nn.Parameter(torch.full((self.num_channels,), default_value))
            param = param.to(device)
            setattr(self, attr, param)
        self._initialized = True

    def parameter_shape(self, x: torch.Tensor) -> tuple | None:
        if hasattr(self, "num_channels"):
            return (1, self.num_channels) + (1,) * (x.ndim - 2)
        return None
