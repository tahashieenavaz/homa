import torch
import torch.nn as nn

from .GraphAttentionHeadModule import GraphAttentionHeadModule


class MultiHeadGraphAttentionModule(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        num_heads: int,
        dropout: float,
        alpha: float,
        concat: bool,
        activation: nn.Module,
        final_activation: nn.Module,
        v2: bool,
        use_layernorm: bool,
    ):
        super().__init__()

        self.concat = concat

        self.heads = nn.ModuleList(
            [
                GraphAttentionHeadModule(
                    input_dimension=input_dimension,
                    output_dimension=output_dimension,
                    dropout=dropout,
                    alpha=alpha,
                    activation=activation,
                    final_activation=final_activation,
                    v2=v2,
                    use_layernorm=use_layernorm,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, features: torch.Tensor, adjacency_matrix: torch.Tensor):
        outputs = [head(features, adjacency_matrix) for head in self.heads]

        if self.concat:
            return torch.cat(outputs, dim=1)
        else:
            return torch.mean(torch.stack(outputs, dim=0), dim=0)
