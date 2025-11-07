import torch
from torchvision.models import swin_v2_b, swin_v2_s, swin_v2_t
from torch.nn.init import kaiming_uniform_ as kaiming


class SwinModule(torch.nn.Module):
    def __init__(self, num_classes: int, variant: str, weights):
        super().__init__()
        self._create_encoder(variant=variant, weights=weights)
        self._create_fc(num_classes=num_classes)

    def variant_instance(self, variant: str):
        variant_map = {"tiny": swin_v2_t, "small": swin_v2_s, "base": swin_v2_b}
        return variant_map.get(variant)

    def _create_encoder(self, variant: str, weights):
        if variant not in ["tiny", "small", "base"]:
            raise ValueError(
                f"Swin variant needs to be one of [tiny, small, base]. Invalid {variant} was provided."
            )
        instance = self.variant_instnace(variant)
        self.encoder = instance(weights=weights)
        self.encoder.head = torch.nn.Identity()

    def _create_fc(self, num_classes: int):
        self.fc = torch.nn.Linear(1024, num_classes)
        kaiming(self.fc.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, images: torch.Tensor):
        features = self.encoder(images)
        return self.fc(features)
