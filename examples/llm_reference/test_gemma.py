import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax import random as jr
from jax.experimental import mesh_utils

from gemma import GemmaDense


def _tiny_kwargs():
  return dict(
    d_model=32,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    head_dim=8,
    d_ff=64,
    local_window=4,
    global_every=2,
    rngs=nnx.rnglib.Rngs(jr.key(0)),
  )


def test_gemma_dense_produces_correct_logit_shape():
  vocab_size, seq_len, batch = 100, 6, 2
  model = GemmaDense(vocab_size, **_tiny_kwargs())
  token_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
  logits, aux_loss = model(token_ids, positions)
  assert logits.shape == (batch, seq_len, vocab_size)
  assert aux_loss == 0.0


def test_gemma_dense_gradients_are_nonzero():
  model = GemmaDense(vocab_size=50, **_tiny_kwargs())
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
def test_gemma_dense_shards_across_2d_mesh():
  # explicit devices=jax.devices()[:4]: create_device_mesh requires the
  # mesh_shape's product to equal the device count exactly, but the
  # skipif above only guarantees >= 4 -- slicing to exactly 4 keeps this
  # test passing under XLA_FLAGS=...device_count=8 too (used by other
  # tests in this suite), not just exactly 4.
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((2, 2), devices=jax.devices()[:4]),
    ("fsdp", "tp"),
  )
  with mesh:
    model = GemmaDense(vocab_size=100, **_tiny_kwargs())
    graphdef, state = nnx.split(model)
    sharding = nnx.get_named_sharding(state, mesh)
    state = jax.device_put(state, sharding)
    nnx.update(model, state)

    # q_proj: TP-sharded column-parallel, ("fsdp", "tp")
    q_kernel = model.blocks[0].attn.q_proj.kernel.value
    assert q_kernel.shape == (32, 32)  # d_model=32, n_heads*head_dim=4*8=32
    assert q_kernel.addressable_shards[0].data.shape == (16, 16)

    # o_proj: TP-sharded row-parallel, ("tp", "fsdp")
    o_kernel = model.blocks[0].attn.o_proj.kernel.value
    assert o_kernel.addressable_shards[0].data.shape == (16, 16)

    token_ids = jnp.zeros((2, 6), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(6), (2, 6))
    logits, aux_loss = model(token_ids, positions)
    assert logits.shape == (2, 6, 100)
