import torch


class GraphAttentionHeadModule(torch.nn.Module):
    def __init__(
        self, input_dimension: int, output_dimension: int, dropout: float, alpha: float
    ):
        super().__init__()
        # feature transform
        self.phi = torch.nn.Linear(input_dimension, output_dimension, bias=False)

        # attention for source
        self.mu = torch.nn.Linear(output_dimension, 1, bias=False)

        # attention for target
        self.xi = torch.nn.Linear(output_dimension, 1, bias=False)

        self.leakyrelu = torch.nn.LeakyReLU(alpha)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, adjacency_matrix: torch.Tensor):
        # transform features
        h = self.phi(features)

        # compute pairwise scores
        e = self.mu(h) + self.xi(h).T
        e = self.leakyrelu(e)
        # masks non-edges
        e = torch.where(adjacency_matrix > 0, e, torch.full_like(e, -9e15))

        # normalize
        a = torch.nn.functional.softmax(e, dim=1)
        a = self.dropout(a)

        # aggregate neighbors
        return torch.nn.functional.elu(a @ h)
