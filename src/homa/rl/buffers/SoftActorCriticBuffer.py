import numpy
import random
import torch
from types import SimpleNamespace
from .Buffer import Buffer


class SoftActorCriticBuffer(Buffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record(
        self,
        state: numpy.ndarray,
        action: int,
        reward: float,
        next_state: numpy.ndarray,
        termination: float,
        probability: numpy.ndarray,
    ):
        self.collection.append(
            (state, action, reward, next_state, termination, probability)
        )

    def sample(self, k: int, as_tensor: bool = False):
        batch = random.sample(self.collection, k)
        states, actions, rewards, next_states, terminations, probabilities = zip(*batch)

        states = numpy.array(states)
        actions = numpy.array(actions)
        rewards = numpy.array(rewards)
        next_states = numpy.array(next_states)
        terminations = numpy.array(terminations)
        probabilities = numpy.array(probabilities)

        # add one dimension to both rewards and terminations
        rewards = numpy.expand_dims(rewards, axis=-1)
        terminations = numpy.expand_dims(terminations, axis=-1)

        if as_tensor:
            states = torch.from_numpy(states).float()
            actions = torch.from_numpy(actions).float()
            rewards = torch.from_numpy(rewards).float()
            next_states = torch.from_numpy(next_states).float()
            terminations = torch.from_numpy(terminations).float()
            probabilities = torch.from_numpy(probabilities).float()

        return SimpleNamespace(
            **{
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "terminations": terminations,
                "probabilities": probabilities,
            }
        )

    def sample_torch(self, k: int):
        return self.sample(k=k, as_tensor=True)
