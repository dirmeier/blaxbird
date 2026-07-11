"""Mixtral-style decoder-only transformer: GQA + RoPE + full causal
attention + real sparse top-k expert routing via capacity-based
dispatch/combine einsums (not dense-compute-then-select).

The dispatch/combine formulation lets JAX's SPMD partitioner (GSPMD)
insert the cross-device communication automatically when the expert
axis is sharded -- no hand-written jax.lax.all_to_all. Verified live
before writing this plan (see plan header): sharding the token axis
along one mesh axis and the expert axis along a DIFFERENT mesh axis
produces output numerically identical to the unsharded computation, with
real collective ops (all-gather, all-reduce) present in the compiled
HLO, and this holds with no explicit with_sharding_constraint calls and
no mesh threaded into __call__ -- plain nnx.with_partitioning weight
annotations are sufficient, matching examples/fsdp_tp_demo's pattern.
"""

import jax
from flax import nnx
from jax import numpy as jnp


class SparseMoEFFN(nnx.Module):
  """Mixtral-style top-k-routed mixture-of-experts feed-forward block
  with real capacity-based dispatch/combine (not dense-compute-then-
  select). Expert weights are stacked into single tensors with a leading
  n_experts axis so that axis can be sharded across a mesh's "expert"
  axis."""

  def __init__(
    self, d_model, d_ff, n_experts, n_active, *, rngs, capacity_factor=1.25
  ):
    """Construct a sparse MoE feed-forward block.

    Args:
      d_model: model (residual stream) dimensionality.
      d_ff: hidden (expansion) dimensionality of each expert.
      n_experts: total number of experts.
      n_active: number of experts activated per token (top-k).
      rngs: random keys.
      capacity_factor: per-expert buffer capacity multiplier. Per-expert
        capacity = ceil(capacity_factor * n_active * n_tokens /
        n_experts). Standard Switch-Transformer value is 1.25. Tokens
        beyond an expert's capacity in a batch are dropped (their
        contribution from that slot is zeroed, not misrouted).
    """
    self.n_experts = n_experts
    self.n_active = n_active
    self.capacity_factor = capacity_factor
    self.router = nnx.Linear(d_model, n_experts, use_bias=False, rngs=rngs)

    expert_partitioning = nnx.with_partitioning(
      nnx.initializers.lecun_normal(), ("expert", None, None)
    )
    key = rngs.params()
    k1, k2, k3 = jax.random.split(key, 3)
    self.gate = nnx.Param(
      expert_partitioning(k1, (n_experts, d_model, d_ff))
    )
    self.up = nnx.Param(
      expert_partitioning(k2, (n_experts, d_model, d_ff))
    )
    self.down = nnx.Param(
      expert_partitioning(k3, (n_experts, d_ff, d_model))
    )

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Route tokens to the top-k experts via capacity-based dispatch/
    combine and combine their outputs.

    Args:
      x: input array, shape (batch, seq, d_model).

    Returns:
      a tuple (output, aux_loss): output has the same shape as x;
      aux_loss is a scalar Switch-Transformer-style load-balancing loss.
    """
    b, s, d = x.shape
    flat = x.reshape(b * s, d)
    n_tok = flat.shape[0]
    logits = self.router(flat)
    probs = jax.nn.softmax(logits, axis=-1)
    top_probs, top_idx = jax.lax.top_k(probs, self.n_active)
    top_probs = top_probs / jnp.sum(top_probs, axis=-1, keepdims=True)

    capacity = int(
      jnp.ceil(self.capacity_factor * self.n_active * n_tok / self.n_experts)
    )

    expert_onehot = jax.nn.one_hot(top_idx, self.n_experts)
    flat_onehot = expert_onehot.reshape(-1, self.n_experts)
    position_in_expert = (
      jnp.cumsum(flat_onehot, axis=0) * flat_onehot - flat_onehot
    )
    position_in_expert = jnp.sum(position_in_expert, axis=-1)
    within_capacity = position_in_expert < capacity
    position_in_expert = position_in_expert.reshape(n_tok, self.n_active)
    within_capacity = within_capacity.reshape(n_tok, self.n_active)

    capacity_onehot = jax.nn.one_hot(
      position_in_expert.astype(jnp.int32), capacity
    )
    dispatch_mask = jnp.sum(
      expert_onehot[..., None]
      * capacity_onehot[:, :, None, :]
      * within_capacity[:, :, None, None],
      axis=1,
    )  # (n_tok, n_experts, capacity)
    combine_weight = jnp.sum(
      expert_onehot[..., None]
      * capacity_onehot[:, :, None, :]
      * within_capacity[:, :, None, None]
      * top_probs[:, :, None, None],
      axis=1,
    )  # (n_tok, n_experts, capacity)

    dispatched = jnp.einsum("td,tec->ecd", flat, dispatch_mask)
    g = jnp.einsum("ecd,edf->ecf", dispatched, self.gate.value)
    u = jnp.einsum("ecd,edf->ecf", dispatched, self.up.value)
    h = jax.nn.silu(g) * u
    expert_out = jnp.einsum("ecf,efd->ecd", h, self.down.value)
    combined = jnp.einsum("ecd,tec->td", expert_out, combine_weight)

    density = jnp.mean(probs, axis=0)
    chosen_mask = jax.nn.one_hot(top_idx, self.n_experts).sum(axis=1)
    chosen_frac = jnp.mean(chosen_mask, axis=0)
    aux_loss = self.n_experts * jnp.sum(density * chosen_frac)

    return combined.reshape(b, s, d), aux_loss
