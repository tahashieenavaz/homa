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
        v2: bool,
    ):
        super().__init__()
        self.v2: bool = v2

        self.phi = torch.nn.Linear(input_dimension, output_dimension, bias=True)

        # separate source, target attentions
        self.mu = torch.nn.Linear(output_dimension, 1, bias=True)
        self.xi = torch.nn.Linear(output_dimension, 1, bias=True)

        # LeakyReLU is a special case because in the original paper, they have explicitly defined
        # alpha = 0.2
        if activation is torch.nn.LeakyReLU:
            self.activation = torch.nn.LeakyReLU(alpha)
        else:
            self.activation = activation()

        self.final_activation = final_activation()
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(output_dimension)

    def forward(self, features: torch.Tensor, adj: torch.Tensor):
        # computes features
        h = self.phi(features)
        h = self.norm(h)

        # computes attention coefficients
        if not self.v2:
            e = self.mu(h) + self.xi(h).T
            e = self.activation(e)
        else:
            scores = h.unsqueeze(1) + h.unsqueeze(0)
            e = self.mu(scores).squeeze(-1)

        e = self.activation(e)
        e = e.masked_fill(adj == 0, float("-inf"))

        a = torch.nn.functional.softmax(e, dim=1)
        a = self.dropout(a)
        return self.final_activation(a @ h)
