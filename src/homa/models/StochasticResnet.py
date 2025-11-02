import torch
from torchvision.models import resnet50
from .utils import replace_relu


class StochasticResnet(torch.nn.Module):
    def __init__(self, outputs: int):
        super(StochasticResnet, self).__init__()
        self._create_encoder()
        self._create_activation_pool()
        self.fc = torch.nn.Linear(2048, outputs)
        replace_relu(self.resnet, self.activation_pool)

    def _create_encoder(self):
        self.encoder = resnet50(weights="DEFAULT")
        self.encoder.fc = torch.nn.Identity()

    def _create_activation_pool(self):
        self.activation_pool = [torch.nn.ReLU()]

    def forward(self, images: torch.Tensor):
        features = self.encoder(images)
        return self.fc(features)
