import torch


class PatchEmbedding(torch.nn.Module):
    def __init__(self, patch_size: int, channels: int, embedding_dimension: int):
        super().__init__()
        self.patch_size: int = patch_size
        self.channels: int = channels
        self.phi = torch.nn.Linear(channels * patch_size**2, embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.channels * self.patch_size**2)
        return self.phi(x)
