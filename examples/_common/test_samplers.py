import pytest
from flax import nnx
from jax import random as jr

from _common import samplers
from _common.parameterizations import EDMConfig


def test_get_sampler_fn_returns_euler():
  fn = samplers.get_sampler_fn("euler")
  assert fn is samplers.euler_sample_fn


def test_get_sampler_fn_returns_heun():
  fn = samplers.get_sampler_fn("heun")
  assert fn is samplers.heun_sample_fn


def test_get_sampler_fn_unknown_name_raises():
  with pytest.raises(ValueError, match="unknown sampler"):
    samplers.get_sampler_fn("not-a-sampler")


class _DummyModel(nnx.Module):
  def __init__(self, *, rngs):
    self.linear = nnx.Linear(4, 4, rngs=rngs)

  def __call__(self, inputs, context, times):
    del context, times
    return self.linear(inputs)


def test_heun_sample_fn_runs():
  model = _DummyModel(rngs=nnx.rnglib.Rngs(jr.key(0)))
  config = EDMConfig(n_sampling_steps=3)
  sample_fn = samplers.get_sampler_fn("heun")(config)
  context = jr.normal(jr.key(1), (2, 4))
  samples = sample_fn(model, jr.key(2), sample_shape=(2, 4), context=context)
  assert samples.shape == (2, 4)
