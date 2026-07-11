"""Train ShardedMLP on random data under a 2D FSDP+TP mesh.

Run with XLA_FLAGS="--xla_force_host_platform_device_count=4" to see real
sharding across 4 simulated CPU devices; runs (unsharded, degenerate)
without it too.
"""

import itertools

import jax
import optax
from flax import nnx
from jax import numpy as jnp
from jax import random as jr
from jax.experimental import mesh_utils

from blaxbird import train_fn
from model import ShardedMLP


def dummy_step(model, rng_key, batch, **kwargs):
  del rng_key, kwargs

  def loss_fn(model):
    return jnp.mean((model(batch["x"]) - batch["y"]) ** 2)

  return nnx.value_and_grad(loss_fn)(model)


def dummy_val(model, rng_key, batch, **kwargs):
  del rng_key, kwargs
  return jnp.mean((model(batch["x"]) - batch["y"]) ** 2)


def run(n_steps: int) -> None:
  n_devices = jax.local_device_count()
  fsdp = 2 if n_devices >= 4 else 1
  tp = n_devices // fsdp
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((fsdp, tp)), ("fsdp", "tp")
  )
  with mesh:
    model = ShardedMLP(64, 256, rngs=nnx.rnglib.Rngs(jr.key(0)))
    optimizer = nnx.Optimizer(model, tx=optax.sgd(1e-2))

    batch = {"x": jnp.ones((16, 64)), "y": jnp.zeros((16, 64))}
    itr = itertools.cycle([batch])

    train = train_fn(
      fns=(dummy_step, dummy_val),
      n_steps=n_steps,
      eval_every_n_steps=max(1, n_steps // 2),
      n_eval_batches=1,
      mesh=mesh,
      data_partition_spec=jax.sharding.PartitionSpec("fsdp"),
    )
    train(jr.key(1), optimizer, itr, itr)
    print("done. up.kernel sharding:", optimizer.model.up.kernel.value.sharding)


if __name__ == "__main__":
  run(n_steps=4)
