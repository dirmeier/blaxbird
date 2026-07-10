import jax.numpy as jnp
from flax import nnx
from jax import random as jr

from _common.nn.unet import AttentionBlock, Downsample, ResBlock, Upsample


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
