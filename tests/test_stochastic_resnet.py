import pytest
import torch
from homa.models import StochasticResnet, Model
from homa.models.modules import ResnetModule


@pytest.fixture
def stochastic_resnet_model():
    return StochasticResnet(num_classes=5, lr=0.001)


def test_resnet_initialization(stochastic_resnet_model):
    assert isinstance(stochastic_resnet_model, StochasticResnet)
    assert isinstance(stochastic_resnet_model, Model)
    assert hasattr(stochastic_resnet_model, "network")
    assert hasattr(stochastic_resnet_model, "optimizer")
    assert hasattr(stochastic_resnet_model, "criterion")
    assert isinstance(stochastic_resnet_model.network, ResnetModule)
    assert isinstance(stochastic_resnet_model.optimizer, torch.optim.SGD)
    assert isinstance(stochastic_resnet_model.criterion, torch.nn.CrossEntropyLoss)
