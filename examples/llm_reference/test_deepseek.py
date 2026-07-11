import jax.numpy as jnp
from flax import nnx
from jax import random as jr

from deepseek import MLAAttention
from layers import make_causal_mask


def _make_mla():
  return MLAAttention(
    d_model=32,
    n_heads=4,
    d_latent=8,
    head_dim_nope=6,
    head_dim_rope=2,
    rngs=nnx.rnglib.Rngs(jr.key(0)),
  )


def test_mla_preserves_shape():
  attn = _make_mla()
  x = jnp.ones((2, 6, 32))
  positions = jnp.broadcast_to(jnp.arange(6), (2, 6))
  mask = make_causal_mask(6)
  out = attn(x, positions, mask)
  assert out.shape == x.shape


def test_mla_causal_mask_blocks_future_positions():
  """Same defining correctness property as GQAAttention: perturbing a
  future token must not change any earlier position's output. This is
  the one place a subtle bug in the decoupled content/RoPE split could
  leak future positions into the past."""
  attn = _make_mla()
  x = jnp.ones((2, 6, 32))
  x_perturbed = x.at[:, -1, :].set(x[:, -1, :] * 100.0)
  positions = jnp.broadcast_to(jnp.arange(6), (2, 6))
  mask = make_causal_mask(6)
  out = attn(x, positions, mask)
  out_perturbed = attn(x_perturbed, positions, mask)
  assert jnp.allclose(out[:, :-1], out_perturbed[:, :-1], atol=1e-5)
