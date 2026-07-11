import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax import random as jr
from jax.experimental import mesh_utils

from deepseek import DeepSeekMLA, MLAAttention
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


def _tiny_kwargs():
  return dict(
    d_model=32,
    n_layers=4,
    n_heads=4,
    d_latent=8,
    head_dim_nope=6,
    head_dim_rope=2,
    d_ff=64,
    rngs=nnx.rnglib.Rngs(jr.key(0)),
  )


def test_deepseek_mla_produces_correct_logit_shape():
  vocab_size, seq_len, batch = 100, 6, 2
  model = DeepSeekMLA(vocab_size, **_tiny_kwargs())
  token_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
  logits, aux_loss = model(token_ids, positions)
  assert logits.shape == (batch, seq_len, vocab_size)
  assert aux_loss == 0.0


def test_deepseek_mla_gradients_are_nonzero():
  model = DeepSeekMLA(vocab_size=50, **_tiny_kwargs())
  token_ids = jnp.zeros((2, 6), dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(6), (2, 6))

  def loss_fn(model):
    logits, aux_loss = model(token_ids, positions)
    return jnp.mean(logits**2) + aux_loss

  grads = nnx.grad(loss_fn)(model)

  leaves = jax.tree_util.tree_leaves(grads)
  assert all(jnp.any(leaf != 0) for leaf in leaves if leaf.size > 0)


@pytest.mark.skipif(
  jax.local_device_count() < 4,
  reason="needs XLA_FLAGS=--xla_force_host_platform_device_count=4",
)
def test_deepseek_mla_shards_across_2d_mesh():
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((2, 2)), ("fsdp", "tp")
  )
  with mesh:
    model = DeepSeekMLA(vocab_size=100, **_tiny_kwargs())
    graphdef, state = nnx.split(model)
    sharding = nnx.get_named_sharding(state, mesh)
    state = jax.device_put(state, sharding)
    nnx.update(model, state)

    up_k_kernel = model.blocks[0].attn.up_k.kernel.value
    assert up_k_kernel.shape == (8, 24)  # d_latent=8, n_heads*head_dim_nope=4*6=24
    assert up_k_kernel.addressable_shards[0].data.shape == (4, 12)

    token_ids = jnp.zeros((2, 6), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(6), (2, 6))
    logits, aux_loss = model(token_ids, positions)
    assert logits.shape == (2, 6, 100)
