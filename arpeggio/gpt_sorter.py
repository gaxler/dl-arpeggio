from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Union

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


def build_optimizers(conf: OptimizerConf):
    """
    * Filter Linear layers from everyone else, train linear layer with weight decay.
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    _train_off = lambda tree: jtu.tree_map(lambda _: False, tree)
    _train_on = lambda tree: jtu.tree_map(lambda _: True, tree)

    def _linear_weights(pytree):
        return jtu.tree_leaves(
            jtu.tree_map(
                lambda l: l.weight if is_linear(l) else None, pytree, is_leaf=is_linear
            )
        )

    wd_params = lambda pt: eqx.tree_at(
        _linear_weights, _train_off(pt), replace_fn=lambda _: True
    )
    no_wd_params = lambda pt: eqx.tree_at(
        _linear_weights, _train_on(pt), replace_fn=lambda _: False
    )

    optim = optax.chain(
        optax.clip_by_global_norm(conf.clip_grad_norm),
        optax.masked(
            optax.adam(learning_rate=conf.lr, b1=conf.adam_beta1, b2=conf.adam_beta2),
            no_wd_params,
        ),
        optax.masked(
            optax.adamw(learning_rate=conf.wd_lr, weight_decay=conf.weight_decay),
            wd_params,
        ),
    )
    return optim


@dataclass
class LoggingConf:
    steps_per_epoch: int = 1000

    def epoch_end(self, steps: int) -> bool:
        return steps > 0 and steps % self.steps_per_epoch == 0


class GPTTrainer:
    @classmethod
    def build_gpt_and_optimizers(
        cls,
        gpt_conf: GPTConf,
        opt_conf: OptimizerConf,
        rng_key: "jax.random.PRNGKey",
        log: LoggingConf = None,
    ):
        # model & optimizer
        rng_key, gpt_init_key = jrandom.split(rng_key, 2)
        gpt = GPT(conf=gpt_conf, key=gpt_init_key)
        opt = build_optimizers(opt_conf)

        # computing the loss & gradients
        @eqx.filter_value_and_grad
        def compute_loss(model, x, mask, keys):
            logits = jax.vmap(model)(x, keys)
            loss_vals = jax.vmap(jax.vmap(single_sample_xent))(
                logits[:, :-1, :], x[:, 1:]
            )
            unnormed = (loss_vals * mask).sum()

            return unnormed / mask.sum()

        # forward pass: comp loss and grads -> compute optimizer updates -> update model weights.
        @eqx.filter_jit
        def step(model, x, mask, keys, param_update_fn, opt_state):
            loss, grads = compute_loss(model, x, mask, keys)
            updates, new_opt_state = param_update_fn(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return loss, model, new_opt_state

        if log is None:
            log = LoggingConf()

        return cls(
            model=gpt,
            optimizer=opt,
            step_fn=step,
            log_conf=log,
            rng_key=rng_key,
        )

    def __init__(
        self,
        model: eqx.Module,
        optimizer: optax.GradientTransformation,
        step_fn: Callable,
        log_conf: LoggingConf,
        rng_key: jax.random.PRNGKey,
    ) -> None:

        self.model = model
        self.opt_state = optimizer.init(self.model)
        self.grad_update = optimizer.update

        self._step_fn = step_fn

        self._steps = 0
        self.ema_logs = EMACollection(decay=0.33)
        self.log_events = log_conf
        self._cur_rng = rng_key

    def rng_key(
        self, num_keys: int = 1
    ) -> Union[jax.random.PRNGKey, Sequence[jax.random.PRNGKey]]:
        keys = jrandom.split(self._cur_rng, num=num_keys + 1)
        out, self._cur_rng = keys[:-1], keys[-1]
        if len(out) == 1:
            return out[0]
        return out

    def log_loss(self, loss_val):
        ema_loss = self.ema_logs.add(float(loss_val), "loss")

        if self.log_events.epoch_end(self._steps):
            print(f"[{self._steps:05d}] Loss: {ema_loss:.4f}")

        return

    def step(self, *loss_fn_inp):

        loss_val, model, new_state = self._step_fn(
            self.model, *loss_fn_inp, self.grad_update, self.opt_state
        )

        self._steps += 1
        self.log_loss(loss_val)

        self.model = model
        self.opt_state = new_state

        return self.log_events.epoch_end(self._steps)

    def predict(self, tokens: Array) -> Tuple[Array, Array]:
        if tokens.ndim > 1:
            bsize = tokens.shape[0]
            res = jax.vmap(self.model)(tokens, self.rng_key(num_keys=bsize))
        else:
            res = self.model(tokens, self.rng_key())

        pred = np.argmax(res, axis=-1)

        return res, pred

    def _gen(self, idxs):
        return self.model.generate(
            idxs, max_pred_tokens=idxs.shape[0], key=None  # self._rng_key()
        )

    def gen_from_tokens(self, promt_idxs: Int[Array, "prompt_len"]):
        idxs = self._gen(promt_idxs)
        return idxs.tolist()

    def gen_from_promt(self, prompt: str) -> str:
        promt_idxs = jnp.array(sorting_task.txt_encode(prompt))
        idxs = self._gen(promt_idxs)
        return sorting_task.token_decode(idxs.tolist())


@dataclass
class TrainerConf:
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

    trainer = GPTTrainer.build_gpt_and_optimizers(
        gpt_conf=trainer_conf.gpt,
        opt_conf=trainer_conf.optimizers,
        rng_key=prng_key,
        log=trainer_conf.logging,
    )

    dataloader = trainer_conf.task.get_dataloader(batch_size=trainer_conf.batch_size)
    sorting_task = trainer_conf.task

    for epoch_idx in range(trainer_conf.num_epochs):

        for batch in dataloader:
            rng_keys = trainer.rng_key(num_keys=trainer_conf.batch_size)
            is_epoch_end = trainer.step(batch.tokens, batch.loss_mask, rng_keys)
            if is_epoch_end:
                break

        print(f"Epoch: {epoch_idx+1}:")
        print("Rows:\n\tOut of dist Seq | GT Training Seq | Predicted Training Seq ")
        prompts = ["3 7 5 ->", "11 5 17 7 ->", "11 5 17 7 13 19 ->"]
        gen_tokens = [
            trainer.gen_from_tokens(sorting_task.txt_encode(txt)) for txt in prompts
        ]
        ood_res = "\n\t".join([sorting_task.token_decode(toks) for toks in gen_tokens])

        batch: SampleBatch = next(dataloader)
        tokens = batch.tokens[0, : batch.seq_len]
        gt_txt = sorting_task.token_decode(tokens.tolist())
        # send the unsorted + arrow as promt
        gen_from_gt = trainer.gen_from_tokens(tokens[: batch.values_in_seq + 1])

        print(f"\t{ood_res}\n\n\t{gt_txt}\n\t{gen_from_gt}")

    return trainer.model


if __name__ == "__main__":
    SEED = 1337
    prng_key = jrandom.PRNGKey(SEED)

    trainer_conf = TrainerConf.from_toml("base.toml")

    gpt = train(trainer_conf, prng_key)


