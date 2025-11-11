import torch


class FeedForwardModule(torch.nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int):
        super().__init__()
        self.feedforward_dimension: int = int(hidden_dimension * 1.618)
        self.embedding = torch.nn.Linear(input_dimension, hidden_dimension)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.zeros_(self.embedding.bias)

        self.block = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dimension),
            torch.nn.Linear(hidden_dimension, self.feedforward_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(self.feedforward_dimension, hidden_dimension),
        )

    def forward(self, x):
        psi = self.embedding(x)
        return psi + self.block(psi)
