import torch
from homa.vision import StochasticSwin
from homa.ensemble import Ensemble


def test_stochastic_swin_ensemble():
    ensemble = Ensemble()
    for _ in range(3):
        model = StochasticSwin(num_classes=10)
        ensemble.record(model.network)

    images = torch.randn(10, 3, 224, 224)
    labels = torch.randint(0, 9, (10,))
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    accuracy = ensemble.accuracy(dataloader)
    assert isinstance(accuracy, float)
