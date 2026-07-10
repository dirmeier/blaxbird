import itertools

import jax.numpy as jnp
import optax
from flax import nnx
from jax import random as jr

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
