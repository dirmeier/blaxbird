import jax.numpy as jnp
import pytest
from flax import nnx
from jax import random as jr

from _common.nn.dit import DiT


def _make_dit(n_classes=None):
  return DiT(
    image_size=(8, 8, 3),
    n_hidden_channels=16,
    patch_size=4,
    n_layers=1,
    n_heads=2,
    n_embedding_features=16,
    n_classes=n_classes,
    rngs=nnx.rnglib.Rngs(jr.key(0)),
  )


def test_unconditional_dit_runs_without_context():
  model = _make_dit(n_classes=None)
  inputs = jnp.ones((2, 8, 8, 3))
  times = jnp.array([0.1, 0.2])
  out = model(inputs, times, context=None)
  assert out.shape == inputs.shape


def test_conditional_dit_runs_with_context():
  model = _make_dit(n_classes=5)
  inputs = jnp.ones((2, 8, 8, 3))
  times = jnp.array([0.1, 0.2])
  context = jnp.array([0, 3])
  out = model(inputs, times, context=context)
  assert out.shape == inputs.shape


def test_conditional_dit_changes_output_per_class():
  model = _make_dit(n_classes=5)
  inputs = jnp.ones((1, 8, 8, 3))
  times = jnp.array([0.1])
  out_class_0 = model(inputs, times, context=jnp.array([0]))
  out_class_1 = model(inputs, times, context=jnp.array([1]))
  assert not jnp.allclose(out_class_0, out_class_1)


def test_missing_context_with_n_classes_raises():
  model = _make_dit(n_classes=5)
  inputs = jnp.ones((2, 8, 8, 3))
  times = jnp.array([0.1, 0.2])
  with pytest.raises(ValueError, match="context"):
    model(inputs, times, context=None)


def test_unexpected_context_without_n_classes_raises():
  model = _make_dit(n_classes=None)
  inputs = jnp.ones((2, 8, 8, 3))
  times = jnp.array([0.1, 0.2])
  with pytest.raises(ValueError, match="context"):
    model(inputs, times, context=jnp.array([0, 1]))
