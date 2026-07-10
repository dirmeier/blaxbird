import jax.numpy as jnp
from flax import nnx
from jax import random as jr

from _common.edm import edm
from _common.parameterizations import EDMConfig


def test_edm_default_config_constructs_without_error():
  train_step, val_step, sample_fn = edm(EDMConfig())
  assert callable(train_step)
  assert callable(val_step)
  assert callable(sample_fn)


class _DummyModel(nnx.Module):
  def __init__(self, *, rngs):
    self.linear = nnx.Linear(4, 4, rngs=rngs)

  def __call__(self, inputs, context, times):
    del context, times
    return self.linear(inputs)


def test_edm_train_step_runs():
  model = _DummyModel(rngs=nnx.rnglib.Rngs(jr.key(0)))
  train_step, _, _ = edm(EDMConfig())
  batch = {"inputs": jnp.ones((2, 4))}
  loss, grads = train_step(model, jr.key(1), batch)
  assert loss.shape == ()


def test_edm_returns_objective_fns():
  from blaxbird._src._types import ObjectiveFns

  fns = edm(EDMConfig())
  assert isinstance(fns, ObjectiveFns)
  assert fns.sample_fn is fns[2]
