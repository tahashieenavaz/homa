import torch


class GraphAttentionHeadModule(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        dropout: float,
        alpha: float,
        activation: torch.nn.Module,
        final_activation: torch.nn.Module,
    ):
        super().__init__()
        # feature transform
        self.phi = torch.nn.Linear(input_dimension, output_dimension, bias=True)

        # attention for source
        self.mu = torch.nn.Linear(output_dimension, 1, bias=True)

        # attention for target
        self.xi = torch.nn.Linear(output_dimension, 1, bias=True)

        # use alpha only if the activation function is leaky relu to replicate the original paper
        if activation == torch.nn.LeakyReLU:
            self.activation = torch.nn.LeakyReLU(alpha)
        else:
            self.activation = activation()

        self.final_activation = final_activation()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, adjacency_matrix: torch.Tensor):
        h = self.phi(features)

        e = self.mu(h) + self.xi(h).T
        e = self.activation(e)
        # masks non-edges
        e = torch.where(adjacency_matrix > 0, e, torch.full_like(e, -9e15))

        a = torch.nn.functional.softmax(e, dim=1)
        a = self.dropout(a)

        return self.final_activation(a @ h)
