from ast import Tuple
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree
from typing import Sequence

from optax._src.base import GradientTransformation

params_only = lambda t: eqx.filter(t, eqx.is_inexact_array, replace=None)
zeros_like = lambda p: jnp.zeros_like(p, dtype=jax.dtypes.canonicalize_dtype(p.dtype))


@dataclass
class AdamState:
    steps: int
    first_moment: PyTree
    second_moment: PyTree


def init_moments(model: PyTree, moments: Sequence[int]) -> Sequence[PyTree]:

    model = params_only(model)
    return tuple(jax.tree_util.tree_map(zeros_like, model) for _ in moments)


def ema_moment(grads: PyTree, ema: PyTree, decay: float, moment: int) -> PyTree:
    """
    Takes PyTrees for gradient and ema of gradients and updates the EMA with current gradient
    """
    update_moment = lambda g, x: decay * x + (1 - decay) * (g**moment)
    return jax.tree_util.tree_map(update_moment, grads, ema)


def build_adam(b1: float, b2: float, eps: float = 1e-8):
    """
    Made this to be compatible with optax. I don't want to re-build the whole of optax, only the interesting parts (that is parts that do actual optimization)
    Returns the EMA gradients scaled by the EMA second moment of gradients.

    You can see here that the each parameter get its own learning rate. This learning rate is $\frac{1}{\sqrt{\nu^2}}$
    """

    def init_adam(model: PyTree) -> AdamState:
        fst, snd = init_moments(model, moments=(1, 2))
        return AdamState(steps=0, first_moment=fst, second_moment=snd)

    def adam_update_fn(
        grads: PyTree, state: AdamState, cur_params: PyTree = None
    ) -> Tuple[PyTree, AdamState]:
        mu = ema_moment(grads, state.first_moment, b1, moment=1)
        nu = ema_moment(grads, state.second_moment, b2, moment=2)

        # bias correction
        bcor1 = 1 - b1**state.steps
        mu_ = jax.tree_map(lambda x: x / bcor1, mu)
        bcor2 = 1 - b2**state.steps
        nu_ = jax.tree_map(lambda x: x / bcor2, nu)

        update_fn = lambda m, v: m / (jnp.sqrt(v) + eps)
        updates = jax.tree_util.tree_map(update_fn, mu_, nu_)
        return updates, AdamState(
            steps=state.steps + 1, first_moment=mu, second_moment=nu
        )

    return GradientTransformation(init=init_adam, update=adam_update_fn)


if __name__ == "__main__":
    init_moments({})
