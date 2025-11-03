import pytest
from homa.models import Resnet, Model


@pytest.fixture
def resnet_model():
    model = Resnet(5, 0.001)
    return model


def test_resnet_initialization():
    assert isinstance(resnet_model, Resnet)
    assert isinstance(resnet_model, Model)
    assert hasattr(resnet_model, "network")
    assert hasattr(resnet_model, "optimizer")
    assert hasattr(resnet_model, "criterion")
    assert isinstance(resnet_model.network, ResnetModule)
    assert isinstance(resnet_model.optimizer, SGD)
    assert isinstance(resnet_model.criterion, nn.CrossEntropyLoss)
