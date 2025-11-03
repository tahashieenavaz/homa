import torch
from torchvision.models import resnet50
from torch.nn.init import kaiming_uniform_ as kaiming


class ResnetModule(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self._create_encoder()
        self._create_fc()

    def _create_encoder(self):
        self.encoder = resnet50(weights="DEFAULT")
        self.encoder.fc = torch.nn.Identity()

    def _create_fc(self):
        self.fc = torch.nn.Linear(2048, self.num_classes)
        kaiming(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, images: torch.Tensor):
        features = self.encoder(images)
        return self.fc(features)
