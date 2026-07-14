"""Tiny_shakespeare data loader.

Tokenization is raw bytes (vocab_size=256). Each TFDS split is a
text blob which is chunked by the loader it into fixed-length
windows.
"""

import grain
import numpy as np
import tensorflow_datasets as tfds
from jax import numpy as jnp
from jax import random as jr

VOCAB_SIZE = 256


def _load_byte_ids(data_dir, split):
  ds = tfds.load(
    "tiny_shakespeare", try_gcs=False, split=split, data_dir=data_dir
  )
  text = next(iter(tfds.as_numpy(ds)))["text"]
  return np.frombuffer(text, dtype=np.uint8)


def data_loaders(
  rng_key,
  data_dir,
  *,
  seq_len=128,
  batch_size=8,
  splits=("train", "validation"),
):
  itrs = []
  for split in splits:
    itr_key, rng_key = jr.split(rng_key)
    byte_ids = _load_byte_ids(data_dir, split).astype(np.int32)
    chunk_len = seq_len + 1
    n_chunks = byte_ids.shape[0] // chunk_len
    chunks = byte_ids[: n_chunks * chunk_len].reshape(n_chunks, chunk_len)

    max_int32 = jnp.iinfo(jnp.int32).max
    seed = int(jr.randint(itr_key, shape=(), minval=0, maxval=max_int32))
    itr = iter(
      grain.MapDataset.source(chunks)
      .shuffle(seed=seed)
      .repeat()
      .batch(batch_size, drop_remainder=True)
      .map(lambda x: {"token_ids": x[:, :-1], "target_ids": x[:, 1:]})
    )
    itrs.append(itr)
  return itrs
