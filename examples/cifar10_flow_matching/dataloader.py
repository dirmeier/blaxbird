import grain
import numpy as np
import tensorflow_datasets as tfds
from jax import numpy as jnp
from jax import random as jr


def data_loaders(
  rng_key,
  outfolder,
  *,
  batch_size=128,
  shuffle=True,
  split="train",
):
  datasets = tfds.data_source(
    "cifar10",
    try_gcs=False,
    split=split,
    data_dir=outfolder,
    builder_kwargs={"file_format": "array_record"},
  )
  if isinstance(split, str):
    datasets = [datasets]
  itrs = []
  if isinstance(shuffle, bool):
    shuffle = [shuffle]
  assert len(datasets) == len(shuffle)
  for dataset, shuffle_me in zip(datasets, shuffle):
    itr_key, rng_key = jr.split(rng_key)
    itr = as_iterable(itr_key, dataset, batch_size, shuffle_me)
    itrs.append(itr)
  return itrs


def as_iterable(rng_key, dataset, batch_size, shuffle):
  def process_fn(example):
    img = example["image"].astype(np.float32) / 255.0
    img = 2.0 * img - 1.0
    return {"inputs": img, "context": example["label"]}

  max_int32 = jnp.iinfo(jnp.int32).max
  seed = int(jr.randint(rng_key, shape=(), minval=0, maxval=max_int32))
  ds = grain.MapDataset.source(dataset).map(process_fn)
  if shuffle:
    ds = ds.shuffle(seed=seed)
  ds = ds.repeat().batch(batch_size, drop_remainder=True)
  return iter(ds)
