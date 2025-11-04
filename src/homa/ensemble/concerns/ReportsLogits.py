import torch
from .errors import PanicsWithoutNetwork


class ReportsLogits(PanicsWithoutNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        self.panic_if_no_network()
        batch_size = x.shape[0]
        logits = torch.zeros((batch_size, self.num_classes))
        for state_dict in self.state_dicts:
            self.network.load_state_dict(state_dict)
            logits += self.network(x)
        return logits

    @torch.no_grad()
    def logits_(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)
