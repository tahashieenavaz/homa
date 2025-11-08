import torch
from .modules import SoftActorModule


class SoftActor:
    def __init__(
        self,
        state_dimension: int,
        hidden_dimension: int,
        action_dimension: int,
        lr: float,
        weight_decay: float,
        alpha: float,
    ):
        self.alpha: float = alpha

        self.network = SoftActorModule(
            state_dimension=state_dimension,
            hidden_dimension=hidden_dimension,
            action_dimension=action_dimension,
        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self, states: torch.Tensor, critic_network: torch.nn.Module):
        self.optimizer.zero_grad()
        loss = self.loss(states=states, critic_network=critic_network)
        loss.backward()
        self.optimizer.step()

    def loss(
        self, states: torch.Tensor, critic_network: torch.nn.Module
    ) -> torch.Tensor:
        actions, probabilities = self.sample(states=states)
        q_alpha, q_beta = critic_network(states, actions)
        q = torch.min(q_alpha, q_beta)
        return (self.alpha * probabilities - q).mean()

    def sample(self, state: torch.Tensor):
        mean, std = self.network(state)
        distribution = torch.distributions.Normal(mean, std)

        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)

        probabilities = distribution.log_prob(pre_tanh).sum(dim=1, keepdim=True)
        correction = torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        return action, probabilities - correction
