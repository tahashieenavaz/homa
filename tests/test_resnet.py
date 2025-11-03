import pytest
import torch
from homa.models import Resnet, Model
from homa.models.modules import ResnetModule


@pytest.fixture
def resnet_model():
    model = Resnet(num_classes=5, lr=0.001)
    return model


def test_resnet_initialization(resnet_model):
    assert isinstance(resnet_model, Resnet)
    assert isinstance(resnet_model, Model)
    assert hasattr(resnet_model, "network")
    assert hasattr(resnet_model, "optimizer")
    assert hasattr(resnet_model, "criterion")
    assert isinstance(resnet_model.network, ResnetModule)
    assert isinstance(resnet_model.optimizer, torch.optim.SGD)
    assert isinstance(resnet_model.criterion, torch.nn.CrossEntropyLoss)
