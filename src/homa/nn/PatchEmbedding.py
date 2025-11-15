import torch


class PatchEmbedding(torch.nn.Module):
    def __init__(
        self, image_size: int, patch_size: int, channels: int, embedding_dimension: int
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = torch.nn.Conv2d(
            channels, embedding_dimension, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
