from dataclasses import dataclass, field
from typing import Iterable, Union, Sequence
from transformers import (
    DataCollatorWithPadding,
    GPT2TokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from jaxtyping import Int, Array, Float
from datasets import load_dataset

import jax.numpy as jnp

HFTokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]


class HFDataset:
    """
    This is a lazy placeholder since datasets doesn't expose a dataset type for annoation

    TODO: make a real type (when introducing type checks)
    """

    pass


@dataclass
class SampleBatch:
    tokens: Int[Array, "bs seq_len"]
    loss_mask: Float[Array, "bs seq_len"]
    batch_size: int
    seq_len: int


def batcher_from_dataset(dataset, tokenizer, batch_size: int) -> Iterable[SampleBatch]:

    token_ds = dataset.map(
        lambda ex: tokenizer(ex["text"], padding=True), batch_size=True
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")

    while True:
        ds = token_ds.shuffle()
        ds = iter(ds)
        batch = []
        for ex in ds:
            batch.append(ex)
            if len(batch) >= batch_size:
                collated = collator(batch)

                yield SampleBatch(
                    tokens=jnp.array(collated["input_ids"]),
                    loss_mask=jnp.array(collated["attention_mask"], dtype=jnp.float32),
                    batch_size=batch_size,
                    seq_len=collated["input_ids"].shape[1],
                )
                batch = []


@dataclass
class HuggingFaceTask:
    """
    :param vocab_size:
    """

    tokenizer: HFTokenizer
    dataset: HFDataset
    vocab_size: int

    @classmethod
    def gpt_tokenizer_from_dataset(cls, name_or_path: str, **other_load_dataset_params):
        """
        This uses HF load_dataset
        """
        dataset = load_dataset(name_or_path, **other_load_dataset_params)

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "[PAD]"

        vocab_size = tokenizer.vocab_size

        return cls(tokenizer, dataset, vocab_size)

    def get_dataloader(self, batch_size: int) -> Iterable[SampleBatch]:
        return batcher_from_dataset(
            dataset=self.dataset, tokenizer=self.tokenizer, batch_size=batch_size
        )
    
    def txt_enc(self, txt: str) -> Sequence[int]:
        return self.tokenizer.encode(txt)
    
    def token_decode(self, tokens: Sequence[int]) -> str:
        if self.tokenizer.pad_token_id:
            tokens = [t for t in tokens if t != self.tokenizer.pad_token_id]
        return self.tokenizer.decode(tokens) 

if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("rotten_tomatoes", split="train")
    bathcer = batcher_from_dataset(dataset=dataset, batch_size=32)

    pass
