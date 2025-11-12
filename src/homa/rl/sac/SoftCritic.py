from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from torch.nn.functional import mse_loss as mse
from .modules import SoftCriticModule
from ..utils import soft_update
from ...common.concerns import MovesModulesToDevice

if TYPE_CHECKING:
    from .SoftActor import SoftActor


class SoftCritic(MovesModulesToDevice):
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        lr: float,
        weight_decay: float,
        gamma: float,
    ):
        self.gamma: float = gamma

        self.zeta = SoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )
        self.eta = SoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )
        self.zeta_target = SoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )
        self.eta_target = SoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )

        # copy source to target when initiated
        self.zeta_target.load_state_dict(self.zeta.state_dict())
        self.eta_target.load_state_dict(self.eta.state_dict())

        self.move_modules()

        self.optimizer = torch.optim.AdamW(
            list(self.zeta.parameters()) + list(self.eta.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        next_states: torch.Tensor,
        actor: SoftActor,
        alpha: float,
    ):
        self.zeta.train()
        self.eta.train()

        self.optimizer.zero_grad()
        loss = self.loss(
            states=states,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            next_states=next_states,
            actor=actor,
            alpha=alpha,
        )
        loss.backward()
        self.optimizer.step()

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        next_states: torch.Tensor,
        actor: torch.nn.Module,
        alpha: float,
    ):
        q_zeta = self.zeta(states, actions)
        q_eta = self.eta(states, actions)
        target = self.calculate_target(
            rewards=rewards,
            terminations=terminations,
            next_states=next_states,
            actor=actor,
            alpha=alpha,
        )
        return mse(q_zeta, target) + mse(q_eta, target)

    @torch.no_grad()
    def calculate_target(
        self,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        next_states: torch.Tensor,
        actor: SoftActor,
        alpha: float,
    ):
        termination_mask = 1 - terminations
        next_actions, next_probabilities = actor.sample(next_states)
        q_zeta_target = self.zeta_target(next_states, next_actions)
        q_eta_target = self.eta_target(next_states, next_actions)
        q_target = torch.min(q_zeta_target, q_eta_target)
        entropy_q = q_target - alpha * next_probabilities
        return rewards + self.gamma * termination_mask * entropy_q

    def update(self, tau: float):
        soft_update(network=self.zeta, target=self.zeta_target, tau=tau)
        soft_update(network=self.eta, target=self.eta_target, tau=tau)
