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
        middle_activation_function: torch.nn.Module,
        middle_amplification_location: str,
        middle_amplification: bool,
        head_amplification: bool,
        middle_activation: bool,
    ):
        super().__init__()
        self.middle_activation: bool = middle_activation
        self.middle_activation_function: bool = middle_activation_function()
        self.middle_amplification: bool = middle_amplification
        self.middle_amplification_location: str = middle_amplification_location

        self.theta = MultiHeadGraphAttentionModule(
            input_dimension=input_dimension,
            output_dimension=hidden_dimension,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            concat=concat,
            activation=activation,
            final_activation=final_activation,
            head_amplification=head_amplification,
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
            head_amplification=head_amplification,
        )
        self.dropout = torch.nn.Dropout(dropout)

        if middle_amplification:
            self.coefficients = torch.nn.Parameter(
                torch.zeros(1, hidden_dimension * num_heads, requires_grad=True)
            )

    def forward(self, features: torch.Tensor, adjacency_matrix: torch.Tensor):
        features = self.dropout(features)
        features = self.theta(features, adjacency_matrix)
        features = self.dropout(features)

        if self.middle_amplification and self.middle_amplification_location == "before":
            features *= self.coefficients.exp()

        if self.middle_activation:
            features = self.middle_activation_function(features)

        if self.middle_amplification and self.middle_amplification_location == "after":
            features *= self.coefficients.exp()

        features = self.sigma(features, adjacency_matrix)
        return torch.nn.functional.log_softmax(features, dim=1)
