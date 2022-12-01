"""

Toy task for transformer exploration: Given a sequence of intergers $\in [0..B]$ generate a sorted versoin of the intergers.

e.g.:
    3 64 8 4 0 24 -> 0 3 4 8 24 64 <EOS>

that is the -> indicates that now its time to start a sorted outoput
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, Tuple
from jaxtyping import Array, Int, Float
import jax.random as jrandom
import numpy as np
import jax
import jax.numpy as jnp

# T is max possible sequnce lenght
@dataclass
class SampleBatch:
    tokens: Int[Array, "bs T"]
    loss_mask: Float[Array, "bs T"]
    values_in_seq: int
    seq_len: int
    """ values in seq + 2 additional tokens "->" and <EOS> """


@dataclass
class SortingTask:
    max_value: int
    """ highest value possible in a sequence """
    max_seq_len: int
    """ maximum length of an unsorted sequence """
    max_batch_seq_len: int = field(init=False)
    """ max possible sequnce lenght. all batches are padded to this lenght """
    vocab_size: int = field(init=False)
    """ number of token in our dictionary. this is max value+1 and the pad, eos and arrow tokens """
    random_seq_ratio: float = 1.0

    def __post_init__(self):
        max_val = self.max_value
        self.token_to_symb = {
            (max_val + 1): "->",
            (max_val + 2): "<PAD>",
            (max_val + 3): "<EOS>",
        }
        self.token_to_symb.update({i: str(i) for i in range(max_val + 1)})

        self.symb_to_token = {v: k for k, v in self.token_to_symb.items()}

        # Sorted+unsoeres+ "->"" token + <EOS> token
        self.max_batch_seq_len = self.max_seq_len * 2 + 2
        self.vocab_size = len(self.symb_to_token)

    def txt_encode(self, txt: str) -> Sequence[int]:
        return [self.symb_to_token[s] for s in txt.split(" ")]

    def token_decode(self, tokens: Sequence[int]) -> str:
        return " ".join([self.token_to_symb[v] for v in tokens])

    def get_dataloader(
        self, batch_size: int, random_seq_len: float = 1.0
    ) -> Iterable[SampleBatch]:
        """
        :param random_seq_len: Ratio of maximum sequence lenght that will be use to generate shorter sequences if needed.
        """
        seq_len = self.max_seq_len
        if random_seq_len < 1:
            seq_len = np.random.randint(
                int(random_seq_len * self.max_seq_len), self.max_seq_len
            )

        max_batch_seq_len = self.max_batch_seq_len

        while True:
            batch_vals = np.random.randint(0, self.max_value, (batch_size, seq_len))
            sorted_bvals = np.sort(batch_vals.copy(), axis=1)

            # create array with <EOS> at the end of a sequence and pad tokens after that. <PAD> tokens
            batch_array = np.empty((batch_size, max_batch_seq_len), dtype=np.int32)
            batch_array.fill(self.symb_to_token["<PAD>"])
            # put the at the end of the sequence
            # " UNSORTED... -> SORTED... <EOS>"
            batch_array[:, :seq_len] = batch_vals
            batch_array[:, seq_len] = self.symb_to_token["->"]
            batch_array[:, seq_len + 1 : (2 * seq_len + 1)] = sorted_bvals
            batch_array[:, 2 * seq_len + 1] = self.symb_to_token["<EOS>"]

            # We'll be predicting the next token out of the current one. So logits & labels will be shifted. This is why the mask is shorter
            # our prediction need to start from "->" and end by the last value in a sequnece. this value will predict <EOS>.
            loss_mask = np.zeros((batch_size, max_batch_seq_len - 1), dtype=np.float32)

            # we need to predict the next token so, mask is shifted by 1 to the left
            # start from "->" and predict the next values.
            loss_mask[:, seq_len : 2 * seq_len + 1] = 1
            loss_mask = jnp.array(loss_mask, dtype=jnp.float32)
            out = jnp.array(batch_array, dtype=jnp.int32)
            yield SampleBatch(
                tokens=out,
                loss_mask=loss_mask,
                values_in_seq=seq_len,
                seq_len=2 * seq_len + 2,
            )
