import jax.numpy as jnp
from flax import nnx
from jax import random as jr

from layers import (
  GQAAttention,
  GeGLU,
  RMSNorm,
  apply_rope,
  make_causal_mask,
  repeat_kv,
  rope_freqs,
)


def test_apply_rope_preserves_shape():
  x = jnp.ones((2, 6, 4, 8))
  positions = jnp.broadcast_to(jnp.arange(6), (2, 6))
  out = apply_rope(x, positions, rope_freqs(8))
  assert out.shape == x.shape


def test_repeat_kv_broadcasts_heads():
  x = jnp.ones((2, 6, 2, 4))
  out = repeat_kv(x, n_rep=3)
  assert out.shape == (2, 6, 6, 4)


def test_make_causal_mask_full_is_lower_triangular():
  mask = make_causal_mask(4)
  expected = jnp.array(
    [
      [True, False, False, False],
      [True, True, False, False],
      [True, True, True, False],
      [True, True, True, True],
    ]
  )
  assert jnp.array_equal(mask, expected)


def test_make_causal_mask_window_restricts_to_local():
  mask = make_causal_mask(4, window=2)
  # position 3 may attend to positions 2,3 only (window=2, excludes 0,1)
  assert jnp.array_equal(mask[3], jnp.array([False, False, True, True]))


def test_rmsnorm_preserves_shape():
  norm = RMSNorm(32, rngs=nnx.rnglib.Rngs(jr.key(0)))
  x = jnp.ones((2, 6, 32))
  assert norm(x).shape == x.shape


def test_geglu_preserves_shape():
  mlp = GeGLU(32, 64, rngs=nnx.rnglib.Rngs(jr.key(0)))
  x = jnp.ones((2, 6, 32))
  assert mlp(x).shape == x.shape


def test_gqa_preserves_shape_under_global_mask():
  d_model, n_heads, n_kv_heads, head_dim, seq_len = 32, 4, 2, 8, 6
  attn = GQAAttention(
    d_model, n_heads, n_kv_heads, head_dim, rngs=nnx.rnglib.Rngs(jr.key(0))
  )
  x = jnp.ones((2, seq_len, d_model))
  positions = jnp.broadcast_to(jnp.arange(seq_len), (2, seq_len))
  mask = make_causal_mask(seq_len)
  out = attn(x, positions, mask)
  assert out.shape == x.shape


def test_local_and_global_masks_produce_different_outputs():
  d_model, n_heads, n_kv_heads, head_dim, seq_len = 32, 4, 2, 8, 6
  attn = GQAAttention(
    d_model, n_heads, n_kv_heads, head_dim, rngs=nnx.rnglib.Rngs(jr.key(0))
  )
  x = jnp.ones((2, seq_len, d_model))
  positions = jnp.broadcast_to(jnp.arange(seq_len), (2, seq_len))
  out_global = attn(x, positions, make_causal_mask(seq_len))
  out_local = attn(x, positions, make_causal_mask(seq_len, window=3))
  assert not jnp.allclose(out_global, out_local)


def test_causal_mask_blocks_future_positions():
  """The defining correctness property of causal attention: changing a
  future token must not change any earlier position's output."""
  d_model, n_heads, n_kv_heads, head_dim, seq_len = 32, 4, 2, 8, 6
  attn = GQAAttention(
    d_model, n_heads, n_kv_heads, head_dim, rngs=nnx.rnglib.Rngs(jr.key(0))
  )
  x = jnp.ones((2, seq_len, d_model))
  x_perturbed = x.at[:, -1, :].set(x[:, -1, :] * 100.0)
  positions = jnp.broadcast_to(jnp.arange(seq_len), (2, seq_len))
  mask = make_causal_mask(seq_len)
  out = attn(x, positions, mask)
  out_perturbed = attn(x_perturbed, positions, mask)
  assert jnp.allclose(out[:, :-1], out_perturbed[:, :-1], atol=1e-5)
