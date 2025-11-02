import torch
from torchvision.models import resnet50
from ..utils import replace_relu
from ...activation import StochasticActivation
from torch.nn.init import kaiming_normal_ as kaiming


class StochasticResnet(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(StochasticResnet, self).__init__()
        self.num_classes = num_classes
        self._create_encoder()
        self._create_fc()
        replace_relu(self.encoder, torch.nn.ReLU, StochasticActivation)

    def _create_encoder(self):
        self.encoder = resnet50(weights="DEFAULT")
        self.encoder.fc = torch.nn.Identity()

    def _create_fc(self):
        self.fc = torch.nn.Linear(2048, self.num_classes)
        kaiming(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, images: torch.Tensor):
        features = self.encoder(images)
        return self.fc(features)
