import jax.numpy as jnp
from flax import nnx
from jax import random as jr

from blaxbird._src.hooks import get_ema_hook


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
  """trainer.py calls every hook as h(step, model=, optimizer=,
  metrics=) -- hook_fn must accept and ignore the kwargs it doesn't
  use."""
  model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(0)))
  optimizer_stub = object()
  hook_fn, _ = get_ema_hook(model, decay=0.9)
  hook_fn(0, model=model, optimizer=optimizer_stub, metrics={"train/loss": 1.0})
