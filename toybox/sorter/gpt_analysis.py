import math
from einops import rearrange
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from arpeggio.transformers.model import (
    GPT,
    CausalAttention,
    TransformerBlock,
    TransformerMLP,
)


from gpt_sorter import TrainerConf


def embed(model: GPT, idxs):
    inp_seq_len = idxs.shape[0]
    pos_idxs = jnp.arange(0, inp_seq_len)

    tok_emb = model.token_emb[idxs]
    pos_emb = model.pos_emb[pos_idxs]
    return tok_emb, pos_emb


def apply_attn(attn: CausalAttention, x):
    kqv_mat = jax.vmap(attn.kqv_enc)(x)
    k, q, v = jnp.split(kqv_mat, 3, axis=-1)

    split_to_heads = lambda mat: rearrange(mat, "t (d h) -> h t d", h=attn.n_heads)
    k, q, v = split_to_heads(k), split_to_heads(q), split_to_heads(v)
    mask = jnp.tril(jnp.ones((1, q.shape[1], k.shape[1])))

    qk_inner_prod = jnp.einsum("hqd,hkd->hqk", q, k)
    logits = qk_inner_prod / math.sqrt(1 / k.shape[-1])
    logits = jnp.where(mask, logits, -jnp.inf)
    attn_w = jax.nn.softmax(logits, axis=-1)

    headed_vals = jnp.einsum("hqK,hKd->hqd", attn_w, v)

    headed_vals = rearrange(headed_vals, "h q d -> q (h d)")

    # project head values over temporal dim
    values = jax.vmap(attn.head_proj)(headed_vals)

    return values, (k, q, v, logits, attn_w)


def apply_mlp(mlp: TransformerMLP, x):
    """Don't have a good reason to open-up the MLP"""
    return jax.vmap(mlp)(x)


def apply_block(block: TransformerBlock, x):
    _x = block.ln_attn(x)

    dx_attn, attn_internals = apply_attn(block.attn, _x)
    x = x + dx_attn

    _x = block.ln_mlp(x)
    dx_mlp = apply_mlp(block.mlp, _x)
    x = x + dx_mlp
    return x, (dx_attn, dx_mlp), attn_internals


def _unzip(lst_of_tups):
    lst_of_lst = [[] for _ in lst_of_tups[0]]
    for tup in lst_of_tups:
        for idx, v in enumerate(tup):
            lst_of_lst[idx].append(v)
    return lst_of_lst


def gpt_step(model: GPT, x):
    tok_emb, pos_emb = embed(model, x)
    x_0 = tok_emb + pos_emb

    x = x_0
    dxs = []
    attn_internals = []  # (k,q,v, logits, attn_w)

    for block in model.blocks:
        x, dx_attn_mlp, _attn_intr = apply_block(block, x)
        dxs.append(dx_attn_mlp)
        attn_internals.append(_attn_intr)

    x = model.final_ln(x)
    logits = jax.vmap(model.classifier)(x)

    return logits, (x_0, _unzip(dxs), _unzip(attn_internals)), (tok_emb, pos_emb)


if __name__ == "__main__":
    SEED = 1337
    trainer_conf = TrainerConf.from_toml("base.toml")

    sorting_task = trainer_conf.task
    dataloader = sorting_task.get_dataloader(batch_size=2)
    base_key = jrandom.PRNGKey(SEED)

    base_key, gpt_init_key = jrandom.split(base_key, 2)
    gpt = GPT(conf=trainer_conf.gpt, key=gpt_init_key)

    batch = next(dataloader)

    (
        logits,
        (x_0, (dx_attn, dx_mlp), (k, q, v, attn_logits, attn_w)),
        (emb_tok, emb_pos),
    ) = gpt_step(gpt, batch.tokens[0])
