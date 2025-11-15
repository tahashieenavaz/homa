import torch
from .PatchEmbedding import PatchEmbedding


class VisionTransformerEncoder(torch.torch.nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        input_channels: int,
        embedding_dimension: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            channels=input_channels,
            embedding_dimension=embedding_dimension,
        )
        num_patches = self.patch_embedding.num_patches
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embedding_dimension))
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, num_patches + 1, embedding_dimension)
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=num_heads,
            dim_feedforward=int(embedding_dimension * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = torch.nn.LayerNorm(embedding_dimension)
        self._init()

    def _init(self):
        torch.nn.init.normal_(self.cls_token, std=1e-6)
        torch.nn.init.normal_(self.pos_embed, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embedding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        features = self.encoder(x)
        features = self.norm(features[:, 0])
        return features
