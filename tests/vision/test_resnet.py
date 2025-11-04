import pytest
import torch
from homa.vision import Resnet, Model
from homa.vision.modules import ResnetModule
from homa import get_device


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


def test_reports_accuracy(resnet_model):
    x = torch.randn(10, 3, 84, 84).to(get_device())
    y = torch.randint(0, 3, (10, 1)).to(get_device())
    accuracy = resnet_model.accuracy(x, y)
    assert isinstance(accuracy, float)

    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    accuracy = resnet_model.accuracy(dataloader)
    assert isinstance(accuracy, float)
