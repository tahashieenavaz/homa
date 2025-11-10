from __future__ import annotations

import numpy
import torch
from typing import TYPE_CHECKING
from .modules import SoftActorModule
from ...core.concerns import MovesModulesToDevice
from ...device import get_device

if TYPE_CHECKING:
    from .SoftCritic import SoftCritic


class SoftActor(MovesModulesToDevice):
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        lr: float,
        weight_decay: float,
        alpha: float,
        min_std: float,
        max_std: float,
    ):
        self.alpha: float = alpha

        self.network = SoftActorModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
            min_std=min_std,
            max_std=max_std,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.move_modules()

    def train(self, states: torch.Tensor, critic: SoftCritic):
        self.network.train()
        self.optimizer.zero_grad()
        loss = self.loss(states=states, critic=critic)
        loss.backward()
        self.optimizer.step()

    def loss(self, states: torch.Tensor, critic: SoftCritic) -> torch.Tensor:
        actions, probabilities = self.sample(states)
        q_zeta = critic.zeta(states, actions)
        q_eta = critic.eta(states, actions)
        q = torch.min(q_zeta, q_eta)
        return (self.alpha * probabilities - q).mean()

    def process_state(
        self, state: numpy.ndarray | torch.Tensor, move_to_device: bool = True
    ) -> torch.Tensor:
        if isinstance(state, numpy.ndarray):
            state = torch.from_numpy(state).float()

        if state.ndim < 2:
            state = state.unsqueeze(0)

        if move_to_device:
            state = state.to(get_device())

        return state

    def sample(self, state: numpy.ndarray | torch.Tensor):
        state = self.process_state(state)

        mean, std = self.network(state)

        # following line prevents standard deviations to be negative
        std = std.exp()

        distribution = torch.distributions.Normal(mean, std)
        z = distribution.rsample()
        action = torch.tanh(z)
        log_probability = distribution.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_probability = log_probability.sum(dim=1, keepdim=True)
        return action, log_probability
