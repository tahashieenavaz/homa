import torch
from torch.nn.functional import mse_loss as mse
from .modules import DualSoftCriticModule
from typing import Type


class SoftCritic:
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        lr: float,
        weight_decay: float,
        tau: float,
        gamma: float,
    ):
        self.tau: float = tau
        self.gamma: float = gamma

        self.network = DualSoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )
        self.target = DualSoftCriticModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        next_states: torch.Tensor,
        actor: torch.nn.Module,
    ):
        self.optimizer.zero_grad()
        loss = self.loss(
            states=states,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            next_states=next_states,
            actor=actor,
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
    ):
        q_alpha, q_beta = self.target(states, actions)
        target = self.calculate_target(
            rewards=rewards,
            terminations=terminations,
            next_states=next_states,
            actor=actor,
        )
        return mse(q_alpha, target) + mse(q_beta, target)

    @torch.no_grad()
    def calculate_target(
        self,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        next_states: torch.Tensor,
        actor: torch.nn.Module,
    ):
        next_actions, next_probabilities = actor.sample(next_states)
        q_alpha, q_beta = self.target(next_states, next_actions)
        q = torch.min(q_alpha, q_beta)
        termination_mask = 1 - terminations
        entropy_q = q - self.alpha * next_probabilities * termination_mask
        return rewards + self.gamma * entropy_q

    def soft_update(
        self, network: Type[torch.nn.Module], target: Type[torch.nn.Module]
    ):
        for s, t in zip(network.parameters(), target.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def update(self):
        self.soft_update(self.network.alpha, self.target.alpha)
        self.soft_update(self.network.beta, self.target.beta)
