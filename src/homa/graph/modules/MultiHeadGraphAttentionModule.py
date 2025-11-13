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
        head_amplification: bool,
    ):
        super().__init__()

        self.head_amplification: bool = head_amplification
        self.concat: bool = concat
        self.num_heads: int = num_heads
        self.output_dimension: int = output_dimension

        if head_amplification:
            self.coefficients = torch.nn.Parameter(
                torch.zeros(1, num_heads * output_dimension, requires_grad=True)
            )

        self.heads = torch.nn.ModuleList(
            [
                GraphAttentionHeadModule(
                    input_dimension=input_dimension,
                    output_dimension=output_dimension,
                    dropout=dropout,
                    alpha=alpha,
                    activation=activation,
                    final_activation=final_activation,
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

        if self.head_amplification:
            features *= self.coefficients.exp()

        return features
