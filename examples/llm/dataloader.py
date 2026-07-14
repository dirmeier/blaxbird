"""Tiny_shakespeare data loader.

Tokenization is raw bytes (vocab_size=256). Each TFDS split is a
text blob which is chunked by the loader it into fixed-length
windows.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from jax import numpy as jnp
from jax import random as jr

VOCAB_SIZE = 256


def _load_byte_ids(data_dir, split):
  ds = tfds.load(
    "tiny_shakespeare", try_gcs=False, split=split, data_dir=data_dir
  )
  text = next(iter(tfds.as_numpy(ds)))["text"]
  return tf.io.decode_raw(text, tf.uint8)


def data_loaders(
  rng_key,
  data_dir,
  *,
  seq_len=128,
  batch_size=8,
  buffer_size=1024,
  prefetch_size=1,
  splits=("train", "validation"),
):

  itrs = []
  for split in splits:
    itr_key, rng_key = jr.split(rng_key)
    byte_ids = tf.cast(_load_byte_ids(data_dir, split), tf.int32)
    chunk_len = seq_len + 1
    n_chunks = tf.shape(byte_ids)[0] // chunk_len
    chunks = tf.reshape(byte_ids[: n_chunks * chunk_len], (n_chunks, chunk_len))

    max_int32 = jnp.iinfo(jnp.int32).max
    seed = jr.randint(itr_key, shape=(), minval=0, maxval=max_int32)
    itr = (
      tf.data.Dataset.from_tensor_slices(chunks)
      .repeat()
      .shuffle(buffer_size, reshuffle_each_iteration=True, seed=int(seed))
      .batch(batch_size, drop_remainder=True)
      .map(
        lambda x: {"token_ids": x[:, :-1], "target_ids": x[:, 1:]},
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
      )
      .prefetch(prefetch_size)
      .as_numpy_iterator()
    )
    itrs.append(itr)
  return itrs
