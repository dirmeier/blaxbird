import jax
import jax.numpy as jnp
from flax import nnx
from jax import random as jr

from mixtral import SparseMoEFFN


def test_sparse_moe_preserves_shape_and_returns_scalar_aux_loss():
  moe = SparseMoEFFN(
    d_model=8, d_ff=16, n_experts=4, n_active=2, rngs=nnx.rnglib.Rngs(jr.key(0))
  )
  x = jnp.ones((2, 6, 8))
  out, aux_loss = moe(x)
  assert out.shape == x.shape
  assert aux_loss.shape == ()
  assert aux_loss > 0


def test_sparse_moe_matches_naive_per_token_reference_within_capacity():
  """The correctness check for the whole dispatch/combine mechanism:
  with a generous capacity_factor (no drops), the output must exactly
  match a naive reference that gathers each token's top-k experts
  directly, with no einsum dispatch trick. Verified live before writing
  this plan (see plan header) -- this test locks that verification in as
  a regression test."""
  d_model, d_ff, n_experts, n_active = 8, 16, 4, 2
  moe = SparseMoEFFN(
    d_model, d_ff, n_experts, n_active,
    capacity_factor=10.0,
    rngs=nnx.rnglib.Rngs(jr.key(0)),
  )
  x = jr.normal(jr.key(1), (2, 6, d_model))
  out, aux_loss = moe(x)

  # naive reference: gather each token's top-k experts directly
  flat = x.reshape(-1, d_model)
  logits = flat @ moe.router.kernel.value
  probs = jax.nn.softmax(logits, axis=-1)
  top_probs, top_idx = jax.lax.top_k(probs, n_active)
  top_probs = top_probs / jnp.sum(top_probs, axis=-1, keepdims=True)

  def expert_ffn(x_token, e):
    g = x_token @ moe.gate.value[e]
    u = x_token @ moe.up.value[e]
    h = jax.nn.silu(g) * u
    return h @ moe.down.value[e]

  naive_out = jnp.zeros_like(flat)
  for slot in range(n_active):
    for tok in range(flat.shape[0]):
      e = int(top_idx[tok, slot])
      naive_out = naive_out.at[tok].add(
        top_probs[tok, slot] * expert_ffn(flat[tok], e)
      )
  naive_out = naive_out.reshape(x.shape)

  assert jnp.allclose(out, naive_out, atol=1e-4)


def test_sparse_moe_drops_tokens_cleanly_under_tiny_capacity():
  """Capacity overflow must not crash or corrupt other tokens -- dropped
  tokens simply don't contribute from that slot."""
  moe = SparseMoEFFN(
    d_model=8, d_ff=16, n_experts=4, n_active=2,
    capacity_factor=0.01,
    rngs=nnx.rnglib.Rngs(jr.key(0)),
  )
  x = jr.normal(jr.key(1), (2, 6, 8))
  out, aux_loss = moe(x)
  assert out.shape == x.shape
  assert jnp.all(jnp.isfinite(out))


def test_sparse_moe_router_receives_nonzero_gradient():
  moe = SparseMoEFFN(
    d_model=8, d_ff=16, n_experts=4, n_active=2, rngs=nnx.rnglib.Rngs(jr.key(0))
  )
  x = jnp.ones((2, 6, 8))

  def loss_fn(moe):
    out, aux = moe(x)
    return jnp.mean(out**2) + aux

  grads = nnx.grad(loss_fn)(moe)
  assert jnp.any(grads.router.kernel.value != 0)
