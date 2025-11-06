import pytest
import torch
from homa.activations.learnable import (
    AOAF,
    DPReLU,
    DualLine,
    FReLU,
    LeLeLU,
    PERU,
    PiLU,
    ShiLU,
    StarReLU,
)

ACTIVATION_CLASSES = [
    AOAF,
    DPReLU,
    DualLine,
    FReLU,
    LeLeLU,
    PERU,
    PiLU,
    ShiLU,
    StarReLU,
]


@pytest.mark.parametrize("activation_cls", ACTIVATION_CLASSES)
def test_activation_shape_behavior(activation_cls):
    activation = activation_cls()
    correct_dummy = torch.randn(1, 5, 1, 1)
    wrong_dummy = torch.randn(1, 9, 1, 1)
    try:
        output = activation(correct_dummy)
        assert torch.is_tensor(
            output
        ), f"{activation_cls.__name__} did not return a tensor"
    except Exception as e:
        pytest.fail(f"{activation_cls.__name__} failed on correct dummy: {e}")

    with pytest.raises(Exception):
        _ = activation(wrong_dummy)
