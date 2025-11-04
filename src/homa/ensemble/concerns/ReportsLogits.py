import torch


class ReportsLogits:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def panic_if_no_network(self):
        if self.network is None:
            raise ValueError("An empty ensemble cannot generate logits")

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        self.panic_if_no_network()
        batch_size = x.shape[0]
        logits = torch.zeros((batch_size, self.num_classes))
        for model in self.models:
            logits += model(x)
        return logits
