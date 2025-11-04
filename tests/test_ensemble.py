import pytest
from homa.ensemble import Ensemble
from homa.vision import Resnet


@pytest.fixture
def ensemble():
    return Ensemble()


@pytest.fixture
def resnet():
    return Resnet(lr=0.001, num_classes=10)


def test_ensemble_initialization(ensemble):
    assert isinstance(ensemble, Ensemble)


def test_ensemble_records_models(ensemble, resnet):
    assert ensemble.size == 0
    ensemble.record(resnet)
    assert ensemble.size != 0
    ensemble.append(resnet)
    ensemble.push(resnet)
    ensemble.add(resnet)
    assert ensemble.size == 4
    assert ensemble.length == 4
