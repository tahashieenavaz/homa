import torch


class CalculatesMetricNecessities:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def metric_necessities(self, dataloader):
        all_predictions = []
        all_labels = []
        for x, y in dataloader:
            batch_logits_list = []
            for model in self.models:
                batch_logits_list.append(model(x))
            all_batch_logits = torch.stack(batch_logits_list)
            avg_logits = torch.mean(all_batch_logits, dim=0)
            _, preds = torch.max(avg_logits, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        return all_predictions, all_labels
