import math
from typing import Sequence, Tuple, Union, Callable

import equinox as eqx
from equinox import static_field
import jax
import optax
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange
from jaxtyping import Array, Float, Bool, Int

from .configs import AttentionConf, GPTConf
from .normalizations import LayerNorm


MaybeMask = Union[Bool[Array, "q_len kv_len"], None]


def _prng_split(key: jax.random.PRNGKey, num_splits: int):
    """Split PRNG key or return Nones"""
    if key is None:
        return (None,) * num_splits
    return jrandom.split(key, num_splits)


def gelu(x: Float[Array, "d"]) -> Float[Array, "d"]:
    gauss_const = math.sqrt(2.0 / math.pi)

    gauss_coeff = 1 + jnp.tanh(gauss_const * (x + 0.044715 * jnp.power(x, 3)))
    return 0.5 * x * gauss_coeff


def single_head_attn_weights(
    query: Float[Array, "q_len d"],
    key: Float[Array, "kv_len d"],
    mask: MaybeMask = None,
):
    logits = jnp.einsum("qd,kd->qk", query, key)
    logits = logits / math.sqrt(1 / key.shape[-1])

    if mask is not None:
        logits = jnp.where(mask, logits, -jnp.inf)

    return jax.nn.softmax(logits, axis=-1)


def compute_attn_weights(
    query: Float[Array, "num_heads q_len d"],
    key: Float[Array, "num_heads kv_len d"],
    mask: MaybeMask = None,
) -> Float[Array, "num_head q_len kv_len"]:

    logits = jnp.einsum("hqd,hkd->hqk", query, key)
    logits = logits / math.sqrt(1 / key.shape[-1])

    if mask is not None:
        logits = jnp.where(mask, logits, -jnp.inf)

    return jax.nn.softmax(logits, axis=-1)


class CausalAttention(eqx.Module):
    kqv_enc: eqx.nn.Linear
    head_proj: eqx.nn.Linear
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout

    n_heads: int  = static_field()
    embed_dim: int = static_field()

    def __init__(self, conf: AttentionConf, key: "jax.random.PRNGKey") -> None:
        kqv_key, head_proj_key = jrandom.split(key, num=2)

        # this is a simple matrix multiplication. we gonna VMAP it over the sequence dim when calling the forward
        self.kqv_enc = eqx.nn.Linear(
            in_features=conf.embed_dim, out_features=3 * conf.embed_dim, key=kqv_key
        )
        self.embed_dim = conf.embed_dim

        # this is a simple matrix multiplication. we gonna VMAP it over the sequence dim when calling the forward
        self.head_proj = eqx.nn.Linear(
            in_features=conf.embed_dim, out_features=conf.embed_dim, key=head_proj_key
        )

        self.attn_dropout = eqx.nn.Dropout(p=conf.attn_drop_prob)
        self.resid_dropout = eqx.nn.Dropout(p=conf.resid_drop_prob)

        self.n_heads = conf.num_heads

    def __call__(
        self, x: Float[Array, "T D"], do_key: Union["jax.random.PRNGKey", None] = None
    ) -> Float[Array, "T D"]:
        kqv_mat = jax.vmap(self.kqv_enc)(x)
        k, q, v = jnp.split(kqv_mat, 3, axis=-1)

        split_to_heads = lambda mat: rearrange(mat, "t (d h) -> h t d", h=self.n_heads)
        k, q, v = split_to_heads(k), split_to_heads(q), split_to_heads(v)
        causal_mask = jnp.tril(jnp.ones((q.shape[1], k.shape[1])))

        # [h q k]
        attn_w = jax.vmap(single_head_attn_weights, in_axes=(0, 0, None))(
            q, k, causal_mask
        )

        dropout_active = do_key is not None
        attn_do_key, resid_do_key = _prng_split(do_key, 2)

        if dropout_active:
            attn_w = self.attn_dropout(
                attn_w, key=attn_do_key, inference=dropout_active
            )

        headed_vals = jnp.einsum("hqK,hKd->hqd", attn_w, v)
        headed_vals = rearrange(headed_vals, "h q d -> q (h d)")

        # project head values over temporal dim
        values = jax.vmap(self.head_proj)(headed_vals)
        if dropout_active:
            values = self.resid_dropout(
                values, key=resid_do_key, inference=dropout_active
            )

        return values


class TransformerMLP(eqx.Module):
    proj_up: eqx.nn.Linear
    proj_down: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    dim_up_proj_rate: int = 4

    def __init__(
        self, embed_dim: int, dropout_prob: float, key: "jax.random.PRNGKey"
    ) -> None:
        d, D = embed_dim, embed_dim * self.dim_up_proj_rate
        up_key, down_key = jrandom.split(key, 2)
        self.proj_up = eqx.nn.Linear(in_features=d, out_features=D, key=up_key)
        self.proj_down = eqx.nn.Linear(in_features=D, out_features=d, key=down_key)
        self.dropout = eqx.nn.Dropout(dropout_prob)

    def __call__(
        self,
        x: Float[Array, "t d"],
        do_key: Union["jax.random.PRNGKey", None] = None,
        act_fn: Callable[[Array], Array] = gelu,
    ) -> Float[Array, "t d"]:
        z = self.proj_up(x)
        z = act_fn(z)
        z = self.proj_down(z)

        if do_key is not None:
            z = self.dropout(z, key=do_key, inference=do_key is None)
        return z


class TransformerBlock(eqx.Module):
    ln_attn: LayerNorm
    """ layer norm befor attention """
    ln_mlp: LayerNorm
    """ layer norm befor MLP"""
    attn: CausalAttention
    mlp: TransformerMLP

    def __init__(self, conf: AttentionConf, key: "jax.random.PRNGKey" = None) -> None:
        embed_dim = conf.embed_dim
        self.ln_attn = LayerNorm(shape=embed_dim)
        self.ln_mlp = LayerNorm(shape=embed_dim)
        attn_key, mlp_key = jrandom.split(key, 2)
        self.attn = CausalAttention(conf=conf, key=attn_key)
        self.mlp = TransformerMLP(
            embed_dim=embed_dim, dropout_prob=conf.resid_drop_prob, key=mlp_key
        )

    def __call__(
        self, x: Float[Array, "t d"], key: "jax.random.PRNGKey" = None
    ) -> Float[Array, "t d"]:

        _seq_len = x.shape[0]
        if key is None:
            mlp_keys = attn_key = mlp_key = None
        else:
            attn_key, mlp_key = jrandom.split(key, 2)
            mlp_keys = jrandom.split(mlp_key, num=_seq_len)

        x = x + self.attn(self.ln_attn(x), do_key=attn_key)

        _mlp = jax.vmap(self.mlp, in_axes=(0, 0))

        x = x + _mlp(self.ln_mlp(x), mlp_keys)

        return x


class GPT(eqx.Module):
    token_emb: Float[Array, "num_tokens emb_dim"]
    pos_emb: Float[Array, "max_pos emb_dim"]
    blocks: Sequence[TransformerBlock]
    dropout: eqx.nn.Dropout
    final_ln: LayerNorm
    classifier: eqx.nn.Linear

    embed_dim: int
    max_seq_len: int

    def __init__(self, conf: GPTConf, key: "jax.random.PRNGKey") -> None:

        blcoks_key, emb_key, cls_key = jrandom.split(key, 3)

        self.embed_dim = conf.attention.embed_dim
        self.max_seq_len = conf.max_seq_len

        # initialize embedding matricies
        tok_key, pos_key = jrandom.split(emb_key, 2)
        self.token_emb = jrandom.normal(
            key=tok_key, shape=(conf.vocab_size, self.embed_dim)
        )
        self.pos_emb = jrandom.normal(
            key=pos_key, shape=(self.max_seq_len, self.embed_dim)
        )

        self.final_ln = LayerNorm(self.embed_dim)
        block_keys = jrandom.split(blcoks_key, num=conf.num_blocks)
        self.blocks = [
            TransformerBlock(conf=conf.attention, key=bk) for bk in block_keys
        ]
        self.dropout = eqx.nn.Dropout(p=conf.embed_pdrop)
        self.final_ln = LayerNorm(self.embed_dim)
        self.classifier = eqx.nn.Linear(
            in_features=self.embed_dim, out_features=conf.vocab_size, key=cls_key
        )

    def __call__(
        self, idxs: Int[Array, "inp_seq_len"], key: "jax.random.PRNGKey"
    ) -> Float[Array, "num_tokens"]:
        inp_seq_len = idxs.shape[0]
        pos_idxs = jnp.arange(0, inp_seq_len)

        # JAX has no out of bounds checks. Indexing this way return the last element if sequence is too long
        # TODO: not sure i can just add bound checks here withoug trashing accelerator runs, otherwise JAX would have them. we need to make sure trainig data is to spec.
        tok_emb = self.token_emb[idxs]
        pos_emb = self.pos_emb[pos_idxs]

        do_key, blocks_key = jrandom.split(key=key, num=2)
        x = tok_emb + pos_emb
        x = self.dropout(x, key=do_key, inference=do_key is None)
        for block, bk in zip(
            self.blocks, jrandom.split(blocks_key, num=len(self.blocks))
        ):
            x = block(x, key=bk)

        x = self.final_ln(x)
        x = jax.vmap(self.classifier)(x)
        return x

    def generate(
        self,
        promt_idxs: Int[Array, "prompt_len"],
        max_pred_tokens: int,
        key: jax.random.PRNGKey,
    ) -> Int[Array, "prompt_max_out_len"]:

        # [b t]
        self.max_seq_len
        idxs = promt_idxs
        for _ in range(max_pred_tokens):

            seq_len = idxs.shape[0]
            # In case our sequence is longer than max len, we do a running window of max sequence len
            if seq_len > self.max_seq_len:
                idxs = idxs[-self.max_seq_len:]
                seq_len = idxs.shape[0]

            pos_idxs = jnp.arange(0, seq_len)
            tok_emb = self.token_emb[idxs]
            pos_emb = self.pos_emb[pos_idxs]
            x = tok_emb + pos_emb

            do_key, key = _prng_split(key, 2)
            if do_key is not None:
                x = self.dropout(x, key=do_key)

            for block in self.blocks:
                bk, key = _prng_split(key, 2)
                x = block(x, bk)

            x = self.final_ln(x)
            logits = jax.vmap(self.classifier)(x)

            # [b 1]
            pred_idx = jnp.argmax(logits[-1, :], axis=-1, keepdims=True)
            # [b t+1]
            idxs = jnp.concatenate((idxs, pred_idx))

        return idxs

class GPTTrainer:
    
    def __init__(
        self,
        model: eqx.Module,
        optimizer: optax.GradientTransformation,
        step_fn: Callable,
        log_conf: "LoggingConf",
        rng_key: jax.random.PRNGKey,
    ) -> None:

        self.model = model
        self.opt_state = optimizer.init(self.model)
        self.grad_update = optimizer.update

        self._step_fn = step_fn

        self._steps = 0
        self.ema_logs = EMACollection(decay=log_conf.ema_decay)
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

        pred = jnp.argmax(res, axis=-1)

        return res, pred

    def _gen(self, idxs):
        return self.model.generate(
            idxs, max_pred_tokens=idxs.shape[0], key=None  # self._rng_key()
        )

    def gen_from_tokens(self, promt_idxs: Int[Array, "prompt_len"]):
        idxs = self._gen(jnp.array(promt_idxs))
        return idxs.tolist()
