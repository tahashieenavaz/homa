import pytest
import torch
from homa.vision import StochasticSwin, Model
from homa.vision.modules import SwinModule


@pytest.fixture
def stochastic_swin():
    return StochasticSwin(num_classes=5)


def test_resnet_initialization(stochastic_swin):
    assert isinstance(stochastic_swin, StochasticSwin)
    assert isinstance(stochastic_swin, Model)
    assert hasattr(stochastic_swin, "network")
    assert hasattr(stochastic_swin, "optimizer")
    assert hasattr(stochastic_swin, "criterion")
    assert isinstance(stochastic_swin.network, SwinModule)
    assert isinstance(stochastic_swin.optimizer, torch.optim.AdamW)
    assert isinstance(stochastic_swin.criterion, torch.nn.CrossEntropyLoss)
