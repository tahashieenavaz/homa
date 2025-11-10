import torch
from ...device import get_device


class CalculatesMetricNecessities:
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def metric_necessities(self, dataloader):
        predictions, labels = [], []
        device = get_device()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            sum_logits = None
            for model in self.models:
                model.to(device)
                model.eval()
                logits = model(x)
                sum_logits = logits if sum_logits is None else sum_logits + logits
            batch_predictions = sum_logits.argmax(dim=1)
            predictions.extend(batch_predictions.cpu().numpy())
            labels.extend(y.cpu().numpy())
        return predictions, labels
