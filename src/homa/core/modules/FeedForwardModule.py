import torch


class FeedForwardModule(torch.nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int):
        super().__init__()
        intermediate_dimension: int = int(hidden_dimension * 1.618)
        self.embedding = torch.nn.Linear(input_dimension, hidden_dimension)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.zeros_(self.embedding.bias)

        self.stream = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dimension),
            torch.nn.Linear(hidden_dimension, intermediate_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(intermediate_dimension, hidden_dimension),
        )

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        residual = x
        x = self.stream(x)
        return x + residual
