import torch
from .GraphAttentionHeadModule import GraphAttentionHeadModule


class MultiHeadGraphAttentionModule(torch.nn.Module):
    def __init__(self, num_heads: int, in_features: int, out_features: int, alpha=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_out_features = out_features
        self.heads = torch.nn.ModuleList(
            [
                GraphAttentionHeadModule(in_features, out_features, alpha=alpha)
                for _ in range(num_heads)
            ]
        )

    def forward(
        self, node_features: torch.Tensor, adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        outputs = [head(node_features, adj_matrix) for head in self.heads]
        h_new_concat = torch.cat(outputs, dim=1)
        return h_new_concat
