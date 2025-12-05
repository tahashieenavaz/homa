import jax
import jax.numpy as numpy
from jax import Array


def crelu(x: Array) -> Array:
    return numpy.concatenate([jax.nn.relu(x), jax.nn.relu(-x)], axis=-1)
