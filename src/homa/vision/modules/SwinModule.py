import torch
from torchvision.models import swin_v2_b
from torch.nn.init import kaiming_uniform_ as kaiming


class SwinModule(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self._create_encoder()
        self._create_fc()

    def _create_encoder(self):
        self.encoder = swin_v2_b(weights="DEFAULT")
        self.encoder.head = torch.nn.Identity()

    def _create_fc(self):
        self.fc = torch.nn.Linear(1024, self.num_classes)
        kaiming(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, images: torch.Tensor):
        features = self.encoder(images)
        return self.fc(features)
