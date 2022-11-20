import functools as ft
from dataclasses import dataclass
import time
from typing import Callable, Iterable, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jrandom
import numpy as np
import optax
from jaxtyping import Array, Int

from losses import single_sample_xent
from transformers.configs import GPTConf, build_from_data, flatten_dataclass, dict_map
from transformers.model import GPT

from dataloading.toy_tasks import SortingTask, SampleBatch
from log_utils.collections import EMACollection


@dataclass
class OptimizerConf:
    lr: float
    wd_lr: float
    clip_grad_norm: float = 1.0
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    def build(self):
        """
        * Filter Linear layers from everyone else, train linear layer with weight decay.
        """
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        _train_off = lambda tree: jtu.tree_map(lambda _: False, tree)
        _train_on = lambda tree: jtu.tree_map(lambda _: True, tree)

        def _linear_weights(pytree):
            return jtu.tree_leaves(
                jtu.tree_map(
                    lambda l: l.weight if is_linear(l) else None,
                    pytree,
                    is_leaf=is_linear,
                )
            )

        wd_params = lambda pt: eqx.tree_at(
            _linear_weights, _train_off(pt), replace_fn=lambda _: True
        )
        no_wd_params = lambda pt: eqx.tree_at(
            _linear_weights, _train_on(pt), replace_fn=lambda _: False
        )

        optim = optax.chain(
            optax.clip_by_global_norm(self.clip_grad_norm),
            optax.masked(
                optax.adam(
                    learning_rate=self.lr, b1=self.adam_beta1, b2=self.adam_beta2
                ),
                no_wd_params,
            ),
            optax.masked(
                optax.adamw(learning_rate=self.wd_lr, weight_decay=self.weight_decay),
                wd_params,
            ),
        )
        return optim


@dataclass
class LoggingConf:
    steps_per_epoch: int = 1000
    ema_decay: float = 0.33

    def epoch_end(self, steps: int) -> bool:
        return steps > 0 and steps % self.steps_per_epoch == 0


@eqx.filter_value_and_grad
def compute_loss(model, x, mask, key):
    keys = jrandom.split(key, x.shape[0])
    logits = jax.vmap(model)(x, keys)
    loss_vals = jax.vmap(jax.vmap(single_sample_xent))(logits[:, :-1, :], x[:, 1:])
    unnormed = (loss_vals * mask).sum()

    return unnormed / mask.sum()


@eqx.filter_jit
def step(x, mask, prng_key, model, opt_state, param_update_fn):
    """
    forward pass:
        * comp loss and grads ->
        * compute optimizer updates ->
        * update model weights.
    """
    loss, grads = compute_loss(model, x, mask, prng_key)
    updates, new_opt_state = param_update_fn(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, new_opt_state


def get_device_mapped_step(devices: Sequence["Device"]):
    """
    Version of step function with collective communication on the gradients and the loss values.
    Function is pmapped over the devices passes
    """

    @ft.partial(
        jax.pmap,
        in_axes=(0, 0, 0, 0, 0, None),
        axis_name="devices",
        static_broadcasted_argnums=5,  # the update function for adam and adam weight decay stays the same
        devices=devices,
    )
    def multi_device_step(x, mask, prng_key, model, opt_state, param_update_fn):
        loss, grads = compute_loss(model, x, mask, prng_key)
        grads = jax.lax.pmean(grads, axis_name="devices")
        updates, new_opt_state = param_update_fn(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, new_opt_state

    return multi_device_step


class GPTTrainer:
    @classmethod
    def build_gpt_and_optimizers(
        cls,
        gpt_conf: GPTConf,
        opt_conf: OptimizerConf,
        rng_key: "jax.random.PRNGKey",
        log: LoggingConf = None,
        devices: Sequence["Device"] = jax.devices(),
    ):
        # model & optimizer
        rng_key, gpt_init_key = jrandom.split(rng_key, 2)
        gpt = GPT(conf=gpt_conf, key=gpt_init_key)
        opt = opt_conf.build()
        opt_state = opt.init(gpt)
        grad_update_fn = opt.update

        if log is None:
            log = LoggingConf()

        multi_device = len(devices) > 1
        # func for computing the loss & gradients and updating model weights
        _step_fn = get_device_mapped_step(devices=devices) if multi_device else step
        if multi_device:
            print(f"Multi Device: on {len(devices)} devices...")
            _device_count = jnp.arange(len(devices))
            gpt = jax.pmap(lambda _: gpt)(_device_count)
            opt_state = jax.pmap(lambda _: opt_state)(_device_count)

        return cls(
            model=gpt,
            opt_state=opt_state,
            grad_update_fn=grad_update_fn,
            step_fn=_step_fn,
            log_conf=log,
            rng_key=rng_key,
            num_devices=len(devices),
        )

    def __init__(
        self,
        model: eqx.Module,
        opt_state: optax.OptState,
        grad_update_fn: optax.TransformUpdateFn,
        step_fn: Callable,
        log_conf: LoggingConf,
        rng_key: jax.random.PRNGKey,
        num_devices: int,
    ) -> None:

        self.model = model
        self.opt_state = opt_state
        self.grad_update_fn = grad_update_fn

        self._step_fn = step_fn

        self._steps = 0
        self.ema_logs = EMACollection(decay=log_conf.ema_decay)
        self.log_events = log_conf
        self._cur_rng = rng_key
        self._num_devices = num_devices

    @property
    def local_model(self):
        return (
            self.model
            if self._num_devices == 1
            else jax.tree_map(lambda x: x[0], self.model)
        )

    def rng_key(
        self, num_keys: int = 1
    ) -> Union[jax.random.PRNGKey, Sequence[jax.random.PRNGKey]]:
        keys = jrandom.split(self._cur_rng, num=num_keys + 1)
        out, self._cur_rng = keys[:-1], keys[-1]
        if len(out) == 1:
            return out[0]
        return out

    def log_loss(self, loss_val):
        if self._num_devices > 1:
            loss_val = loss_val[0]

        ema_loss = self.ema_logs.add(float(loss_val), "loss")

        if self.log_events.epoch_end(self._steps):
            print(f"[{self._steps:05d}] Loss: {ema_loss:.4f}")

        return

    def step(self, *loss_fn_inp):

        loss_val, model, new_state = self._step_fn(
            *loss_fn_inp,
            self.model,
            self.opt_state,
            self.grad_update_fn,
        )

        self._steps += 1
        self.log_loss(loss_val)

        self.model = model
        self.opt_state = new_state

        return self.log_events.epoch_end(self._steps)

    def predict(self, tokens: Array) -> Tuple[Array, Array]:
        """
        This function runs on a single device, it will get the model from device 0.
        """
        local_model = self.local_model

        if tokens.ndim > 1:
            bsize = tokens.shape[0]
            res = jax.vmap(local_model)(tokens, self.rng_key(num_keys=bsize))
        else:
            res = local_model(tokens, self.rng_key())

        pred = np.argmax(res, axis=-1)

        return res, pred

    def _gen(self, idxs):
        return self.local_model.generate(
            idxs, max_pred_tokens=idxs.shape[0], key=None  # self._rng_key()
        )

    def gen_from_tokens(self, promt_idxs: Int[Array, "prompt_len"]):
        idxs = self._gen(jnp.array(promt_idxs))
        return idxs.tolist()


@dataclass
class TrainerConf:
    """
    :param batch_size: Per-device batch size. This will be replacted to all available devices
    """

    task: SortingTask
    gpt: GPTConf
    optimizers: OptimizerConf
    batch_size: int
    num_epochs: int
    logging: LoggingConf

    @classmethod
    def from_toml(cls, fpath: str):
        import toml

        with open(fpath, "r") as fp:
            data = toml.load(fp)
            data = dict_map(
                lambda k, v: (None, None) if k.startswith("_") else (k, v), data
            )

        return build_from_data(cls=cls, data=data)

    def to_toml(self, dst_path: str):
        import toml

        data = flatten_dataclass(self)
        with open(dst_path, "w") as fp:
            toml.dump(data, fp)


def train(trainer_conf: TrainerConf, prng_key: jax.random.PRNGKey) -> GPT:

    # in case this runs on a cloudTPU cluster we gonna have 8 TPU cores on each
    # of the machines. Each machine will run this script, so dataloader will be replicated
    # on each of the processes.
    # Does this means we need to generete local_device? My undetastnding is

    # this toy task generates its data. so every JAX process will run this.
    local_devices = jax.local_devices()
    num_local_devices = len(local_devices)

    trainer = GPTTrainer.build_gpt_and_optimizers(
        gpt_conf=trainer_conf.gpt,
        opt_conf=trainer_conf.optimizers,
        rng_key=prng_key,
        log=trainer_conf.logging,
        devices=local_devices,
    )

    batch_size = trainer_conf.batch_size
    dataloader = trainer_conf.task.get_dataloader(
        batch_size=batch_size * num_local_devices,
        random_seq_len=trainer_conf.task.random_seq_ratio,
    )

    def make_per_device_iter(dataloader: Iterable[SampleBatch]):
        def _add_axis(*args):
            for arg in args:
                if num_local_devices > 1:
                    arg = arg.reshape(num_local_devices, batch_size, *arg.shape[1:])
                yield arg

        for batch in dataloader:
            batched_inputs = tuple(_add_axis(batch.tokens, batch.loss_mask))
            yield batched_inputs + (trainer.rng_key(num_local_devices),)

    sorting_task = trainer_conf.task

    def _generation_printout(epoch_num):
        print(f"Epoch: {epoch_num}:")
        print("Rows:\n\tOut of dist Seq | GT Training Seq | Predicted Training Seq ")
        prompts = [
            "37 31 ->",
            "3 7 5 ->",
            "11 5 17 7 ->",
            "37 31 29 23 19 ->",
            "11 5 17 7 13 19 ->",
        ]
        gen_tokens = [
            trainer.gen_from_tokens(sorting_task.txt_encode(txt)) for txt in prompts
        ]
        ood_res = "\n\t".join([sorting_task.token_decode(toks) for toks in gen_tokens])

        batch: SampleBatch = next(dataloader)
        tokens = batch.tokens[0, : batch.seq_len]
        gt_txt = sorting_task.token_decode(tokens.tolist())
        # send the unsorted + arrow as promt
        gen_from_gt = sorting_task.token_decode(
            trainer.gen_from_tokens(tokens[: batch.values_in_seq + 1])
        )

        print(f"\t{ood_res}\n\n\t{gt_txt}\n\t{gen_from_gt}")

    try:
        tic = time.time()
        for epoch_idx in range(trainer_conf.num_epochs):

            for step_inp_args in make_per_device_iter(dataloader):
                # is_epoch_end = trainer.step(batch.tokens, batch.loss_mask, rng_keys)
                is_epoch_end = trainer.step(*step_inp_args)
                if is_epoch_end:
                    break
            if epoch_idx % 30 == 0:
                _generation_printout(epoch_num=epoch_idx + 1)

    except KeyboardInterrupt:
        # This function is ment to run in a notebook
        # Want to be able to interrupt the run and return the lastest weights
        pass
    tok = time.time() - tic
    proced_seq = batch_size*num_local_devices*trainer._steps
    seq_sec = proced_seq / tok
    print(
        f"KeyboardInterrupt: \n\t Returning the model after {proced_seq} sequences"
    )
    print(f"\t Processed {seq_sec:.3f} seq/sec.")
    print("\n------------------------\n")
    _generation_printout(epoch_num=epoch_idx + 1)

    return trainer.model


if __name__ == "__main__":
    SEED = 1337
    prng_key = jrandom.PRNGKey(SEED)

    trainer_conf = TrainerConf.from_toml("base.toml")
    # Always keep base.toml with full config spec
    trainer_conf.to_toml("base.toml")

    gpt = train(trainer_conf, prng_key)
