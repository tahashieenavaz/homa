import torch


class GraphAttentionHeadModule(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = torch.nn.Linear(in_features, out_features, bias=False)
        self.a_1 = torch.nn.Parameter(torch.randn(out_features, 1))
        self.a_2 = torch.nn.Parameter(torch.randn(out_features, 1))

        self.leaky_relu = torch.nn.LeakyReLU(self.alpha)
        self.elu = torch.nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a_1, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a_2, gain=1.414)

    def forward(self, node_features, adj_matrix):
        N = node_features.size(0)
        h_prime = self.W(node_features)
        s1 = torch.matmul(h_prime, self.a_1)
        s2 = torch.matmul(h_prime, self.a_2)
        e = s1 + s2.T
        e = self.leaky_relu(e)
        zero_vec = -9e15 * torch.ones_like(e)
        attention_mask = torch.where(
            adj_matrix > 0, e, zero_vec.to(node_features.device)
        )
        attention_weights = F.softmax(attention_mask, dim=1)
        h_new = torch.matmul(attention_weights, h_prime)
        return self.elu(h_new)
