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
    ):
        super().__init__()
        self.theta = MultiHeadGraphAttentionModule(
            input_dimension, hidden_dimension, num_heads, dropout, alpha, concat
        )
        self.sigma = MultiHeadGraphAttentionModule(
            hidden_dimension * num_heads, output_dimension, 1, dropout, alpha, concat
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, adjacency_matrix: torch.Tensor):
        features = self.dropout(features)
        features = self.theta(features, adjacency_matrix)
        features = self.dropout(features)
        features = self.sigma(features, adjacency_matrix)
        return torch.nn.functional.log_softmax(features, dim=1)
