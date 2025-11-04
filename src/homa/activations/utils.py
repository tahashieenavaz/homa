from typing import Optional, Sequence, Tuple, Type

import torch
from torch import nn


def negative_part(x):
    return torch.minimum(x, torch.zeros_like(x))


def positive_part(x):
    return torch.maximum(x, torch.zeros_like(x))


def as_channel_parameters(parameter: torch.Tensor, x: torch.Tensor):
    shape = [1] * x.dim()
    shape[1] = -1
    return parameter.view(*shape)


def device_compatibility_check(model, x: torch.Tensor):
    for p in model.parameters():
        if p.device != x.device or p.dtype != x.dtype:
            p.data = p.data.to(device=x.device, dtype=x.dtype)


def phi_hat(x, a, lam):
    term_pos = torch.maximum(lam - torch.abs(x - a), torch.zeros_like(x))
    term_neg = torch.minimum(torch.abs(x - (a + 2 * lam)) - lam, torch.zeros_like(x))
    return term_pos + term_neg


def _channels_from_module(module: nn.Module) -> Optional[int]:
    for attr in ("channels", "out_channels", "out_features", "num_features"):
        value = getattr(module, attr, None)
        if isinstance(value, int) and value > 0:
            return int(value)
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
        return int(weight.shape[0])
    return None


def _infer_channels(parent: nn.Module, target_name: str, child: nn.Module) -> int:
    value = _channels_from_module(child)
    if value is not None:
        return value

    children: Sequence[Tuple[str, nn.Module]] = list(parent.named_children())
    for index, (name, _) in enumerate(children):
        if name == target_name:
            break
    else:
        raise ValueError(f"Child {target_name} not found in parent {parent.__class__.__name__}.")

    for _, module in reversed(children[:index]):
        value = _channels_from_module(module)
        if value is not None:
            return value

    value = _channels_from_module(parent)
    if value is not None:
        return value

    raise ValueError(
        f"Could not infer channel count for activation {target_name!r} under parent {parent.__class__.__name__}."
    )


def infer_activation_channels(parent: nn.Module, child_name: str, child: nn.Module) -> int:
    return _infer_channels(parent, child_name, child)


def replace_activation(
    model: nn.Module,
    activation: Type[nn.Module],
    needles: Optional[Sequence[Type[nn.Module]]] = None,
    **activation_kwargs,
) -> None:
    if needles is None:
        needles = (nn.ReLU,)
    needle_types = tuple(needles)

    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, needle_types):
                channels = _infer_channels(parent, name, child)
                try:
                    new_module = activation(channels=channels, **activation_kwargs)
                except TypeError as exc:  # pragma: no cover - defensive branch
                    raise TypeError(
                        f"{activation.__name__} must accept a `channels` keyword argument."
                    ) from exc
                setattr(parent, name, new_module)
