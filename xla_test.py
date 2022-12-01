import jax
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
import functools as ft
from einops import rearrange, repeat

from gpt_sorter import TrainerConf, build_optimizers, SampleBatch
from transformers.model import GPT
from losses import single_sample_xent

import time


@eqx.filter_value_and_grad
def compute_loss(model, x, mask, key):
    keys = jrandom.split(key, x.shape[0])
    logits = jax.vmap(model)(x, keys)
    loss_vals = jax.vmap(jax.vmap(single_sample_xent))(logits[:, :-1, :], x[:, 1:])
    unnormed = jax.lax.psum((loss_vals * mask), axis_name="devices").sum()
    norm = jax.lax.psum(mask, axis_name="devices").sum()
    return unnormed / norm


# forward pass: comp loss and grads -> compute optimizer updates -> update model weights.
@ft.partial(
    jax.pmap,
    in_axes=(0, 0, 0, 0, 0, None),
    axis_name="devices",
    static_broadcasted_argnums=5, # the update function for adam and adam weight decay stays the same
)
def step(model, x, mask, opt_state, prng_key, param_update_fn):
    loss, grads = compute_loss(model, x, mask, prng_key)
    grads = jax.lax.pmean(grads, axis_name="devices")
    updates, new_opt_state = param_update_fn(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, new_opt_state


if __name__ == "__main__":
    

    SEED = 1337
    prng_key = jrandom.PRNGKey(SEED)
    num_devices = jax.device_count()

    trainer_conf: TrainerConf = TrainerConf.from_toml("base.toml")
    model_init_key, prng_key = jrandom.split(prng_key, 2)
    gpt = GPT(trainer_conf.gpt, key=model_init_key)

    dataloader = trainer_conf.task.get_dataloader(batch_size=32 * 8)
    batch: SampleBatch = next(dataloader)
    device_xs = rearrange(batch.tokens, "(dev bs) T -> dev bs T", dev=num_devices)
    device_mask = rearrange(batch.loss_mask, "(dev bs) T -> dev bs T", dev=num_devices)
    opt = build_optimizers(trainer_conf.optimizers)
    opt_state = opt.init(gpt)
    tic = None

    # move the model an optimizer state to the devices 
    model = jax.pmap(lambda _: gpt)(jnp.arange(num_devices))
    opt_state = jax.pmap(lambda _: opt_state)(jnp.arange(num_devices))

    for step_ in range(5):
        print(f"step {step_+1}")
        keys = jrandom.split(prng_key, num_devices + 1)
        device_keys, prng_key = keys[:-1], keys[-1]
        if tic is not None:
            print(time.time() - tic)
        tic = time.time()
        loss, new_model, new_opt_state = step(
            model, device_xs, device_mask, opt_state, device_keys, opt.update
        )
        # model = jax.tree_map(lambda x: x[0], new_model)
        # opt_state = jax.tree_map(lambda x: x[0], new_opt_state)
    # jax.vmap(compute_loss, in_axes=(None, 0, 0, 0))(gpt, device_xs, device_mask, device_keys)

    pass

