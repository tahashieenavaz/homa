import torch
from .GraphAttentionHeadModule import GraphAttentionHeadModule


class MultiHeadGraphAttentionModule(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        num_heads: int,
        dropout: float,
        alpha: float,
        concat: bool,
        activation: torch.nn.Module,
        final_activation: torch.nn.Module,
        v2: bool,
    ):
        super().__init__()

        self.num_heads: int = num_heads
        self.output_dimension: int = output_dimension
        self.concat: bool = concat

        self.heads = torch.nn.ModuleList(
            [
                GraphAttentionHeadModule(
                    input_dimension=input_dimension,
                    output_dimension=output_dimension,
                    dropout=dropout,
                    alpha=alpha,
                    activation=activation,
                    final_activation=final_activation,
                    v2=v2,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, features: torch.Tensor, adjacency_matrix: torch.Tensor):
        features = [head(features, adjacency_matrix) for head in self.heads]

        # aggregate results either by concatenating them or averaging over
        if self.concat:
            features = torch.cat(features, dim=1)
        else:
            features = torch.mean(torch.stack(features), dim=0)

        return features
