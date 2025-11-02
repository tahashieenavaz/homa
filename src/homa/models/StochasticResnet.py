import torch
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from .utils import replace_relu


class StochasticResnet(torch.nn.Module):
    def __init__(self, outputs: int):
        super(StochasticResnet, self).__init__()
        self.encoder = resnet50(ResNet50_Weights)
        self.encoder.fc = torch.nn.Identity()
        self.fc = torch.nn.Linear(2048, outputs)
        replace_relu(self.resnet, self.activation_pool)

    def create_activation_pool(self):
        self.activation_pool = [torch.nn.ReLU()]

    def forward(self, images: torch.Tensor):
        features = self.encoder(images)
        return self.fc(features)
