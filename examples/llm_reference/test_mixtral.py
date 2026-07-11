import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax import random as jr
from jax.experimental import mesh_utils

from mixtral import MixtralSMoE, SparseMoEFFN


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


def _tiny_kwargs():
  return dict(
    d_model=32,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    head_dim=8,
    d_ff=64,
    n_experts=4,
    n_active=2,
    rngs=nnx.rnglib.Rngs(jr.key(0)),
  )


def test_mixtral_smoe_produces_correct_logit_shape_and_nonzero_aux_loss():
  vocab_size, seq_len, batch = 100, 6, 2
  model = MixtralSMoE(vocab_size, **_tiny_kwargs())
  token_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
  logits, aux_loss = model(token_ids, positions)
  assert logits.shape == (batch, seq_len, vocab_size)
  assert aux_loss > 0.0


def test_mixtral_smoe_gradients_are_nonzero():
  """Checks that every parameter LEAF (gate/up/down/router/attention
  weights, each a single stacked (n_experts, ...) tensor, not one leaf
  per expert) receives a nonzero gradient somewhere in it. This is
  whole-tensor liveness, not per-expert liveness: with this seed, 2 of
  the 4 experts actually receive zero dispatched tokens in this small
  batch (confirmed by inspecting dispatch counts) -- capacity_factor
  governs token-dropping when an expert is OVERsubscribed, it has no
  bearing on whether an expert is starved of tokens in the first place,
  so raising it would not change this. A starved expert's own gate/up/
  down slice legitimately gets zero gradient this step; the assertion
  still passes because it checks the whole stacked tensor has some
  nonzero entry, not that every expert does. Detecting per-expert
  starvation would need a per-expert-slice assertion, which this test
  does not attempt -- it exists to catch a structurally dead PARAMETER
  (e.g. an unused projection), not routing imbalance."""
  model = MixtralSMoE(vocab_size=50, **{**_tiny_kwargs(), "n_layers": 1})
  token_ids = jnp.zeros((4, 8), dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(8), (4, 8))

  def loss_fn(model):
    logits, aux_loss = model(token_ids, positions)
    return jnp.mean(logits**2) + aux_loss

  grads = nnx.grad(loss_fn)(model)
  leaves = jax.tree_util.tree_leaves(grads)
  assert all(jnp.any(leaf != 0) for leaf in leaves if leaf.size > 0)


@pytest.mark.skipif(
  jax.local_device_count() < 8,
  reason="needs XLA_FLAGS=--xla_force_host_platform_device_count=8",
)
def test_mixtral_smoe_sharded_output_matches_unsharded_reference():
  """The critical correctness test for real expert-parallel dispatch:
  sharding tokens along one mesh axis and experts along a DIFFERENT mesh
  axis must produce output numerically identical to an unsharded
  reference computation with the same weights and inputs. This is what
  actually proves the dispatch/combine + GSPMD auto-communication claim
  -- shape-only tests cannot catch a token routed to the wrong expert if
  the wrong expert happens to produce same-shaped output. Verified live
  before writing this plan (see plan header): ~1e-9 max diff, with real
  collective ops confirmed present in the compiled HLO."""
  kwargs = {**_tiny_kwargs(), "n_layers": 1}
  token_ids = jnp.zeros((4, 8), dtype=jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(8), (4, 8))

  ref_model = MixtralSMoE(vocab_size=50, **kwargs)
  ref_logits, ref_aux = ref_model(token_ids, positions)
  # ref_logits/ref_aux are already concrete values at this point, so
  # resharding ref_model in place below cannot retroactively affect them
  # -- no need for a separate cloned module.

  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((2, 2, 2)), ("fsdp", "tp", "expert")
  )
  with mesh:
    graphdef, state = nnx.split(ref_model)
    sharding = nnx.get_named_sharding(state, mesh)
    state = jax.device_put(state, sharding)
    nnx.update(ref_model, state)

    sharded_logits, sharded_aux = ref_model(token_ids, positions)

  max_diff = jnp.abs(ref_logits - sharded_logits).max()
  assert max_diff < 1e-3, f"sharded output diverges from reference: {max_diff}"
  assert jnp.allclose(ref_aux, sharded_aux, atol=1e-3)


@pytest.mark.skipif(
  jax.local_device_count() < 8,
  reason="needs XLA_FLAGS=--xla_force_host_platform_device_count=8",
)
def test_mixtral_smoe_shards_expert_axis_across_3d_mesh():
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((2, 2, 2)), ("fsdp", "tp", "expert")
  )
  with mesh:
    model = MixtralSMoE(vocab_size=100, **{**_tiny_kwargs(), "n_layers": 1})
    graphdef, state = nnx.split(model)
    sharding = nnx.get_named_sharding(state, mesh)
    state = jax.device_put(state, sharding)
    nnx.update(model, state)

    gate = model.blocks[0].ffn.gate.value
    assert gate.shape == (4, 32, 64)  # n_experts=4, d_model=32, d_ff=64
    assert gate.addressable_shards[0].data.shape == (2, 32, 64)
