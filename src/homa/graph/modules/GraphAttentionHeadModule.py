import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionHeadModule(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        dropout: float,
        alpha: float,
        activation: nn.Module,
        final_activation: nn.Module,
        v2: bool,
        use_layernorm: bool,
    ):
        super().__init__()
        self.v2: bool = v2

        self.phi = nn.Linear(input_dimension, output_dimension, bias=True)
        self.mu = nn.Linear(output_dimension, 1, bias=False)
        self.xi = nn.Linear(output_dimension, 1, bias=False)

        if activation is nn.LeakyReLU:
            self.att_activation = nn.LeakyReLU(alpha)
        else:
            self.att_activation = nn.LeakyReLU(alpha)  # for e_{ij}

        self.final_activation = final_activation()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dimension) if use_layernorm else nn.Identity()

    def forward(self, features: torch.Tensor, adj: torch.Tensor):
        h = self.phi(features)  # [N, out]
        h = self.norm(h)

        if not self.v2:
            e = self.mu(h) + self.xi(h).T
        else:
            scores = h.unsqueeze(1) + h.unsqueeze(0)  # [N, N, out]
            e = self.mu(scores).squeeze(-1)  # [N, N]

        e = self.att_activation(e)
        e = e.masked_fill(adj == 0, float("-inf"))

        a = F.softmax(e, dim=1)
        a = self.dropout(a)

        h_out = a @ h  # [N, out]

        return self.final_activation(h_out)
