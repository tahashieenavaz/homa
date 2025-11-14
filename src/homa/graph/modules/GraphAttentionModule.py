import torch
from .MultiHeadGraphAttentionModule import MultiHeadGraphAttentionModule


class GraphAttentionModule(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        num_heads: int,
        dropout: float,
        alpha: float,
        concat: bool,
        activation: torch.nn.Module,
        final_activation: torch.nn.Module,
        v2: bool,
        use_layernorm: bool,
    ):
        super().__init__()
        self.theta = MultiHeadGraphAttentionModule(
            input_dimension=input_dimension,
            output_dimension=hidden_dimension,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            concat=concat,
            activation=activation,
            final_activation=final_activation,
            v2=v2,
            use_layernorm=use_layernorm,
        )
        self.sigma = MultiHeadGraphAttentionModule(
            input_dimension=hidden_dimension * num_heads,
            output_dimension=output_dimension,
            num_heads=1,
            dropout=dropout,
            alpha=alpha,
            concat=concat,
            activation=activation,
            final_activation=final_activation,
            v2=v2,
            use_layernorm=use_layernorm,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, adjacency_matrix: torch.Tensor):
        features = self.dropout(features)
        features = self.theta(features, adjacency_matrix)
        features = self.dropout(features)
        features = self.sigma(features, adjacency_matrix)
        return torch.nn.functional.log_softmax(features, dim=1)
