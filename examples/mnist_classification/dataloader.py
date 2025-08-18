import grain.python as grain
import tensorflow_datasets as tfds
from grain._src.core.transforms import Batch
from jax import numpy as jnp
from jax import random as jr


def data_loaders(
  rng_key,
  outfolder,
  *,
  n_epochs=100,
  n_workers=0,
  batch_size=128,
  buffer_size=1,
  shuffle=True,
  split="train",
):
  datasets = tfds.data_source(
    "mnist",
    split=split,
    data_dir=outfolder,
    download_and_prepare_kwargs={"file_format": "parquet"},
  )
  if isinstance(split, str):
    datasets = [datasets]
  itrs = []
  for dataset in datasets:
    itr_key, rng_key = jr.split(rng_key)
    itr = _as_grain_data_loader(
      itr_key, dataset, n_epochs, n_workers, batch_size, buffer_size, shuffle
    )
    itrs.append(itr)
  return itrs


def _as_grain_data_loader(
  rng_key, itr, n_epochs, n_workers, batch_size, buffer_size, shuffle
):
  max_int32 = jnp.iinfo(jnp.int32).max
  seed = jr.randint(rng_key, shape=(), minval=0, maxval=max_int32)

  index_sampler = grain.IndexSampler(
    num_records=len(itr),
    num_epochs=n_epochs,
    shard_options=grain.ShardOptions(
      shard_index=0, shard_count=1, drop_remainder=True
    ),
    shuffle=shuffle,
    seed=int(seed),
  )
  data_loader = grain.DataLoader(
    data_source=itr,
    operations=[Batch(batch_size, drop_remainder=True)],
    shard_options=grain.ShardOptions(
      shard_index=0, shard_count=1, drop_remainder=True
    ),
    sampler=index_sampler,
    worker_count=n_workers,
  )
  return data_loader
