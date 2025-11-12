import torch
from .SoftActorCriticRepository import SoftActorCriticRepository
from ...common.concerns import MovesModulesToDevice


class SoftActorCriticTemperature(MovesModulesToDevice):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        action_dimension: int,
        device: torch.device,
    ):
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=device))
        self.optimizer = torch.optim.Adam(
            [self.log_alpha], lr=lr, weight_decay=weight_decay
        )
        self.target_entropy = -float(action_dimension)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self):
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()

    def loss(self) -> torch.Tensor:
        # updated inside actor
        log_probability = SoftActorCriticRepository.log_probability

        theta = (-log_probability - self.target_entropy).detach()
        return (self.alpha * theta).mean()
