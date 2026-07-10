import jax.numpy as jnp
import pytest
from flax import nnx
from jax import random as jr

from _common.nn.unet import AttentionBlock, Downsample, ResBlock, UNet, Upsample


def test_res_block_changes_channels():
  block = ResBlock(8, 16, 32, rngs=nnx.rnglib.Rngs(jr.key(0)))
  inputs = jnp.ones((2, 8, 8, 8))
  embedding = jnp.ones((2, 32))
  out = block(inputs, embedding)
  assert out.shape == (2, 8, 8, 16)


def test_res_block_same_channels_uses_identity_skip():
  block = ResBlock(8, 8, 32, rngs=nnx.rnglib.Rngs(jr.key(0)))
  assert block.skip is None
  inputs = jnp.ones((2, 8, 8, 8))
  embedding = jnp.ones((2, 32))
  out = block(inputs, embedding)
  assert out.shape == (2, 8, 8, 8)


def test_attention_block_preserves_shape():
  block = AttentionBlock(16, n_heads=4, rngs=nnx.rnglib.Rngs(jr.key(0)))
  inputs = jnp.ones((2, 4, 4, 16))
  out = block(inputs)
  assert out.shape == inputs.shape


def test_downsample_halves_spatial_dims():
  block = Downsample(8, rngs=nnx.rnglib.Rngs(jr.key(0)))
  inputs = jnp.ones((2, 16, 16, 8))
  out = block(inputs)
  assert out.shape == (2, 8, 8, 8)


def test_upsample_doubles_spatial_dims():
  block = Upsample(8, rngs=nnx.rnglib.Rngs(jr.key(0)))
  inputs = jnp.ones((2, 8, 8, 8))
  out = block(inputs)
  assert out.shape == (2, 16, 16, 8)


def _make_unet(n_classes=None):
  return UNet(
    image_size=(32, 32, 3),
    n_hidden_channels=32,
    channel_mults=(1, 2, 2),
    n_res_blocks=2,
    attention_resolutions=(16,),
    n_embedding_features=64,
    n_heads=4,
    n_classes=n_classes,
    rngs=nnx.rnglib.Rngs(jr.key(0)),
  )


def test_unconditional_unet_preserves_input_shape():
  model = _make_unet(n_classes=None)
  inputs = jnp.ones((2, 32, 32, 3))
  times = jnp.array([0.1, 0.5])
  out = model(inputs, times, context=None)
  assert out.shape == inputs.shape


def test_conditional_unet_preserves_input_shape():
  model = _make_unet(n_classes=10)
  inputs = jnp.ones((2, 32, 32, 3))
  times = jnp.array([0.1, 0.5])
  context = jnp.array([0, 3])
  out = model(inputs, times, context=context)
  assert out.shape == inputs.shape


def test_conditional_unet_missing_context_raises():
  model = _make_unet(n_classes=10)
  inputs = jnp.ones((2, 32, 32, 3))
  times = jnp.array([0.1, 0.5])
  with pytest.raises(ValueError, match="context"):
    model(inputs, times, context=None)
