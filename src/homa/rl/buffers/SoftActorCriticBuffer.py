import numpy
import random
import torch
from types import SimpleNamespace
from .Buffer import Buffer
from ...device import get_device


class SoftActorCriticBuffer(Buffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record(
        self,
        state: numpy.ndarray,
        action: numpy.ndarray,
        reward: float,
        next_state: numpy.ndarray,
        termination: float,
    ):
        self.collection.append((state, action, reward, next_state, termination))

    def sample(self, k: int, as_tensor: bool = False, move_to_device: bool = True):
        batch = random.sample(self.collection, k)
        states, actions, rewards, next_states, terminations = zip(*batch)

        states = numpy.array(states, dtype=numpy.float32)
        actions = numpy.array(actions, dtype=numpy.float32)
        rewards = numpy.array(rewards, dtype=numpy.float32)
        next_states = numpy.array(next_states, dtype=numpy.float32)
        terminations = numpy.array(terminations, dtype=numpy.float32)

        # add one dimension to both rewards and terminations
        rewards = numpy.expand_dims(rewards, axis=-1)
        terminations = numpy.expand_dims(terminations, axis=-1)

        if as_tensor:
            states = torch.from_numpy(states).float()
            actions = torch.from_numpy(actions).float()
            rewards = torch.from_numpy(rewards).float()
            next_states = torch.from_numpy(next_states).float()
            terminations = torch.from_numpy(terminations).float()

        if as_tensor and move_to_device:
            _device = get_device()
            states = states.to(_device)
            actions = actions.to(_device)
            rewards = rewards.to(_device)
            next_states = next_states.to(_device)
            terminations = terminations.to(_device)

        return SimpleNamespace(
            **{
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "terminations": terminations,
            }
        )

    def sample_torch(self, k: int):
        return self.sample(k=k, as_tensor=True)
