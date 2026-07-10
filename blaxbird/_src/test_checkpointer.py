import optax
from flax import nnx
from jax import numpy as jnp
from jax import random as jr

from blaxbird._src.checkpointer import get_default_checkpointer


class _Linear(nnx.Module):
  def __init__(self, *, rngs):
    self.linear = nnx.Linear(2, 2, rngs=rngs)

  def __call__(self, x):
    return self.linear(x)


def test_save_and_restore_last_roundtrip_optimizer_only(tmp_path):
  model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(0)))
  optimizer = nnx.Optimizer(model, tx=optax.sgd(1e-2))

  save_fn, _, restore_last_fn = get_default_checkpointer(
    str(tmp_path), save_every_n_steps=1
  )
  save_fn(
    1, model=optimizer.model, optimizer=optimizer, metrics={"val/loss": 0.5}
  )

  new_model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(1)))
  new_optimizer = nnx.Optimizer(new_model, tx=optax.sgd(1e-2))
  # restore_last_fn takes/returns optimizer only -- no separate model arg
  restored_optimizer = restore_last_fn(new_optimizer)
  assert isinstance(restored_optimizer, nnx.Optimizer)
  assert jnp.allclose(
    restored_optimizer.model.linear.kernel.value,
    optimizer.model.linear.kernel.value,
  )


def test_custom_criterion_key_and_best_mode(tmp_path):
  model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(0)))
  optimizer = nnx.Optimizer(model, tx=optax.sgd(1e-2))

  save_fn, restore_best_fn, _ = get_default_checkpointer(
    str(tmp_path),
    save_every_n_steps=1,
    criterion_key="val/accuracy",
    best_mode="max",
  )
  save_fn(
    1,
    model=optimizer.model,
    optimizer=optimizer,
    metrics={"val/accuracy": 0.7},
  )
  save_fn(
    2,
    model=optimizer.model,
    optimizer=optimizer,
    metrics={"val/accuracy": 0.9},
  )

  new_model = _Linear(rngs=nnx.rnglib.Rngs(jr.key(1)))
  new_optimizer = nnx.Optimizer(new_model, tx=optax.sgd(1e-2))
  restored_optimizer = restore_best_fn(new_optimizer)
  assert isinstance(restored_optimizer, nnx.Optimizer)
