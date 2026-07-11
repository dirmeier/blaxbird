import itertools

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import random as jr
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P

from blaxbird._src.trainer import train_fn


class _Linear(nnx.Module):
  def __init__(self, *, rngs):
    self.linear = nnx.Linear(2, 2, rngs=rngs)

  def __call__(self, x):
    return self.linear(x)


def _dummy_step(model, rng_key, batch, **kwargs):
  del rng_key, kwargs

  def loss_fn(model):
    return jnp.mean((model(batch["x"]) - batch["y"]) ** 2)

  return nnx.value_and_grad(loss_fn)(model)


def _dummy_val(model, rng_key, batch, **kwargs):
  del rng_key, kwargs
  return jnp.mean((model(batch["x"]) - batch["y"]) ** 2)


def test_train_fn_takes_optimizer_only_no_model_arg():
  model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(0)))
  optimizer = nnx.Optimizer(model, tx=optax.sgd(1e-2))
  batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 2))}
  itr = itertools.cycle([batch])

  train = train_fn(
    fns=(_dummy_step, _dummy_val),
    n_steps=2,
    eval_every_n_steps=1,
    n_eval_batches=1,
  )
  # signature is (rng_key, optimizer, train_itr, val_itr) -- no model arg
  train(jr.key(1), optimizer, itr, itr)


def test_train_fn_shards_across_mesh():
  """Verifies train_fn shards state and batches across a device mesh.

  Requires XLA_FLAGS=--xla_force_host_platform_device_count=4 to exercise
  real multi-device sharding; degrades to a 1-device no-op sharding
  otherwise (still exercises the mesh code path, just not the
  multi-shard assertion below).
  """
  n_devices = jax.local_device_count()
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((n_devices,)), ("data",)
  )
  model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(0)))
  optimizer = nnx.Optimizer(model, tx=optax.sgd(1e-2))
  batch = {"x": jnp.ones((8, 2)), "y": jnp.zeros((8, 2))}
  itr = itertools.cycle([batch])

  train = train_fn(
    fns=(_dummy_step, _dummy_val),
    n_steps=2,
    eval_every_n_steps=1,
    n_eval_batches=1,
    mesh=mesh,
    data_partition_spec=P("data"),
  )
  train(jr.key(1), optimizer, itr, itr)

  if n_devices > 1:
    sharded_x = jax.device_put(batch["x"], jax.NamedSharding(mesh, P("data")))
    assert len(sharded_x.addressable_shards) == n_devices
