import torch


class ReportsLogits:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = torch.zeros((batch_size, self.num_classes))
        for factory, weight in zip(self.factories, self.weights):
            model = factory(num_classes=self.num_classes)
            model.load_state_dict(weight)
            logits += model(x)
        return logits

    @torch.no_grad()
    def logits_(self, *args, **kwargs):
        return self.logits(*args, **kwargs)
