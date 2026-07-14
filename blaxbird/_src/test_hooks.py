import itertools

import jax.numpy as jnp
import optax
from flax import nnx
from jax import random as jr

from blaxbird._src.hooks import get_ema_hook
from blaxbird._src.trainer import train_fn


class _Linear(nnx.Module):
  def __init__(self, *, rngs):
    self.linear = nnx.Linear(4, 4, rngs=rngs)

  def __call__(self, x):
    return self.linear(x)


def test_ema_hook_tracks_toward_but_lags_current_params():
  model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(0)))
  original = model.linear.kernel.value.copy()
  hook_fn, get_ema_model = get_ema_hook(model, decay=0.9)

  for step in range(3):
    model.linear.kernel.value = model.linear.kernel.value + 1.0
    hook_fn(step, model=model)

  ema_model = get_ema_model(model)
  assert jnp.all(original < ema_model.linear.kernel.value)
  assert jnp.all(ema_model.linear.kernel.value < model.linear.kernel.value)


def test_ema_model_does_not_alias_live_model():
  model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(0)))
  hook_fn, get_ema_model = get_ema_hook(model, decay=0.9)
  hook_fn(0, model=model)

  ema_model = get_ema_model(model)
  kernel_before = ema_model.linear.kernel.value.copy()
  model.linear.kernel.value = model.linear.kernel.value + 100.0
  assert jnp.array_equal(ema_model.linear.kernel.value, kernel_before)


def test_ema_model_is_usable():
  model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(0)))
  hook_fn, get_ema_model = get_ema_hook(model, decay=0.9)
  hook_fn(0, model=model)
  ema_model = get_ema_model(model)
  out = ema_model(jnp.ones((1, 4)))
  assert out.shape == (1, 4)


def test_ema_hook_ignores_extra_trainer_kwargs():
  model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(0)))
  optimizer_stub = object()
  hook_fn, _ = get_ema_hook(model, decay=0.9)
  hook_fn(0, model=model, optimizer=optimizer_stub, metrics={"train/loss": 1.0})


def _dummy_step(model, rng_key, batch, **kwargs):
  del rng_key, kwargs

  def loss_fn(model):
    return jnp.mean((model(batch["x"]) - batch["y"]) ** 2)

  return nnx.value_and_grad(loss_fn)(model)


def _dummy_val(model, rng_key, batch, **kwargs):
  del rng_key, kwargs
  return jnp.mean((model(batch["x"]) - batch["y"]) ** 2)


def test_ema_hook_integrates_with_train_fn():
  model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(0)))
  optimizer = nnx.Optimizer(model, tx=optax.sgd(1e-2))
  hook_fn, get_ema_model = get_ema_hook(model, decay=0.9)

  batch = {"x": jnp.ones((4, 4)), "y": jnp.zeros((4, 4))}
  itr = itertools.cycle([batch])

  train = train_fn(
    fns=(_dummy_step, _dummy_val),
    n_steps=3,
    eval_every_n_steps=1,
    n_eval_batches=1,
    hooks=[hook_fn],
  )
  train(jr.key(1), optimizer, itr, itr)

  ema_model = get_ema_model(optimizer.model)
  assert ema_model.linear.kernel.value.shape == (4, 4)
