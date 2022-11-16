import jax
from jaxtyping import Array, Int, Float


def single_sample_xent(logits: Float[Array, "num_cls"], label: Int) -> Float:
    num_cls = logits.shape[0]
    gt = jax.nn.one_hot(label, num_classes=num_cls)
    loss = -(jax.nn.log_softmax(logits) * gt).sum()
    return loss
