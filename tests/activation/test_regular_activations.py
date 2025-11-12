import pytest
import torch
import homa.activations as activations
import homa


def test_all_activations():
    data = torch.randn(10, 10)

    for name in dir(activations):
        if name.startswith("__"):
            continue
        obj = getattr(activations, name)
        if callable(obj) and obj not in [
            homa.activations.ActivationFunction,
            homa.activations.AdaptiveActivationFunction,
        ]:
            instance = obj()
            try:
                result = instance(data)
                assert isinstance(result, torch.Tensor)
            except Exception as e:
                pytest.fail(f"{name} failed: {e}")
