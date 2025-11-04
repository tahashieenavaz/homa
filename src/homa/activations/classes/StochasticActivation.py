import random
import torch
from torch import nn

from .APLU import APLU
from .GALU import GALU
from .SmallGALU import SmallGALU
from .MELU import MELU
from .WideMELU import WideMELU
from .PDELU import PDELU
from .SReLU import SReLU


class StochasticActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self._gate_cls = random.choice(
            [
                APLU,
                GALU,
                SmallGALU,
                MELU,
                WideMELU,
                PDELU,
                SReLU,
                nn.ReLU,
                nn.PReLU,
                nn.LeakyReLU,
                nn.ELU,
            ]
        )
        self._gates = nn.ModuleDict()

    @staticmethod
    def _channel_key(x: torch.Tensor) -> str:
        if x.ndim < 2:
            return "scalar"
        return str(int(x.shape[1]))

    @staticmethod
    def _move_gate(gate: nn.Module, x: torch.Tensor) -> nn.Module:
        if torch.is_floating_point(x) or torch.is_complex(x):
            return gate.to(device=x.device, dtype=x.dtype)
        return gate.to(device=x.device)

    def _get_gate(self, x: torch.Tensor) -> nn.Module:
        key = self._channel_key(x)
        if key not in self._gates:
            gate = self._move_gate(self._gate_cls(), x)
            self._gates[key] = gate
        gate = self._gates[key]

        param = next(gate.parameters(recurse=True), None)
        if param is not None:
            if param.device != x.device or param.dtype != x.dtype:
                gate = self._move_gate(gate, x)
                self._gates[key] = gate
        else:
            buffer = next(gate.buffers(recurse=True), None)
            if buffer is not None:
                if buffer.device != x.device or buffer.dtype != x.dtype:
                    gate = self._move_gate(gate, x)
                    self._gates[key] = gate

        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._get_gate(x)
        return gate(x)
