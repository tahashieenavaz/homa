import torch
from .GraphAttentionHeadModule import GraphAttentionHead


class GraphAttentionModule(torch.nn.Module):
    def __init__(self, num_heads, in_features, head_out_features, alpha=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_out_features = head_out_features
        self.heads = torch.nn.ModuleList(
            [
                GraphAttentionHead(in_features, head_out_features, alpha=alpha)
                for _ in range(num_heads)
            ]
        )

    def forward(self, node_features, adj_matrix):
        outputs = [head(node_features, adj_matrix) for head in self.heads]
        h_new_concat = torch.cat(outputs, dim=1)
        return h_new_concat
