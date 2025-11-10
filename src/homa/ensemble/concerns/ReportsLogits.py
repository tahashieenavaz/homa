import torch


class ReportsLogits:
    def __init__(self):
        super().__init__()

    def logits_average(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_sim(x) / len(self.factories)

    def logits_sum(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        logits = torch.zeros((batch_size, self.num_classes))
        for factory, weight in zip(self.factories, self.weights):
            model = factory(num_classes=self.num_classes)
            model.load_state_dict(weight)
            logits += model(x)
        return logits

    def check_aggregation_strategy(self, aggregation: str):
        if aggregation not in ["mean", "average", "sum"]:
            raise ValueError(
                f"Ensemble aggregation strategy must be in [mean, average, sum], but found {aggregation}."
            )

    def logits(self, x: torch.Tensor, aggregation: str = "mean") -> torch.Tensor:
        self.check_aggregation_strategy(aggregation=aggregation)
        logits_handlers = {
            "mean": self.logits_average,
            "average": self.logits_average,
            "sum": self.logits_sum,
        }
        handler = logits_handlers.get(aggregation)
        return handler(x)

    @torch.no_grad()
    def logits_(self, *args, **kwargs):
        return self.logits(*args, **kwargs)
