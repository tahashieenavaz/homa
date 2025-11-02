import torch
from torchvision.models import resnet50
from ..utils import replace_modules
from ...activation import StochasticActivation


class StochasticResnet(torch.nn.Module):
    def __init__(self, outputs: int):
        super(StochasticResnet, self).__init__()
        self._create_encoder()
        self.fc = torch.nn.Linear(2048, outputs)
        replace_modules(self.resnet, torch.nn.ReLU, StochasticActivation)

    def _create_encoder(self):
        self.encoder = resnet50(weights="DEFAULT")
        self.encoder.fc = torch.nn.Identity()

    def forward(self, images: torch.Tensor):
        features = self.encoder(images)
        return self.fc(features)
