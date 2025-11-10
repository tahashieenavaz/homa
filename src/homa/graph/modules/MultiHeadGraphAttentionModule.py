import torch
from .GraphAttentionHeadModule import GraphAttentionHead


class MultiHeadGraphAttentionModule(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        num_heads: int,
        dropout: float,
        alpha: float,
        concat: bool,
    ):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                GraphAttentionHead(input_dimension, output_dimension, dropout, alpha)
                for _ in range(num_heads)
            ]
        )
        self.concat: bool = concat

    def forward(self, features: torch.Tensor, adjacency_matrix: torch.Tensor):
        features = [head(features, adjacency_matrix) for head in self.heads]

        # aggregate
        if self.concat:
            features = torch.cat(features, dim=1)
        else:
            features = torch.mean(torch.stack(features), dim=0)

        return features
