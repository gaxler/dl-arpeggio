from typing import Sequence, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


NormShape = Union[int, Sequence[int]]


class LayerNorm(eqx.Module):
    """
    Implementation of Layer Normalization.
    """

    sigma_param: Array
    mu_param: Array
    eps: float = 1e-5  # cargo cult pytorch

    def __init__(self, shape: NormShape) -> None:
        self.mu_param = jnp.zeros(shape)
        self.sigma_param = jnp.ones(shape)

    def __call__(self, x: Array) -> Array:
        mu = jnp.mean(x, keepdims=True)
        sig_sqr = jnp.var(x, keepdims=True)
        inv_std = jax.lax.rsqrt(sig_sqr + self.eps)
        normed = (x - mu) * inv_std

        res = normed * self.sigma_param + self.mu_param

        return res
