"""Qwen3Next-style LM.

Architecture design:
  - Attention: 3:1 linear/standard hybrid. Every linear_every-th layer
    (1-indexed, default 4) is standard GQA attention; the other ~75% are
    Gated DeltaNet linear-attention layers.
    - Gated DeltaNet: linear recurrent sequence mixer via the delta rule
      with data-dependent decay (alpha) and write-rate (beta) gates over
      L2-normalized queries/keys; O(seq) sequential scan.
    - standard attention: GQA with QK-norm, partial RoPE (leading 25% of
      head_dim), and a sigmoid output gate.
  - Positional encoding: partial RoPE on standard-attention layers;
    DeltaNet layers are position-implicit (recurrent).
  - Normalization: RMSNorm (no mean-centering, no bias), pre-norm, with
    a separate norm before the mixer and FFN plus a final norm before
    the head; QK-norm on standard-attention layers.
  - FFN: sparse MoE -- top-k routed experts (capacity-based dispatch/
    combine) plus one always-on shared expert; SwiGLU experts and a
    Switch-style load-balancing aux loss.
  - Embeddings: untied -- a separate input nnx.Embed and output lm_head.
  - Sharding: 3D FSDP+TP+expert (routed-expert axis sharded).

  Faithful to the real model:
  - 3:1 Gated DeltaNet / standard-attention hybrid
  - delta-rule recurrence with decay + write-rate gates, L2-normed Q/K
  - output-gated, QK-normed, partial-RoPE GQA
  - top-k routed MoE with a shared expert

  Divergences from the real model (real / implemented):
  - chunked/parallel-form DeltaNet kernel / an O(seq) sequential scan
  - short causal Conv1D before the recurrence / q/k/v feed it directly
  - dropless routing / Switch-style capacity dropping
  - zero-centered RMSNorm / standard RMSNorm
  - multi-token prediction / single next-token objective
  - illustrative layer sizes, not any real variant's config
"""

import jax
from flax import nnx
from jax import numpy as jnp
from nn.layers import RMSNorm, apply_partial_rope, repeat_kv, rope_freqs, tp_linear


class SparseMoEFFN(nnx.Module):
  def __init__(
    self, din, dhid, n_experts, n_active, *, rngs, capacity_factor=1.25
  ):
    self.n_experts = n_experts
    self.n_active = n_active
    self.capacity_factor = capacity_factor
    self.router = nnx.Linear(din, n_experts, use_bias=False, rngs=rngs)

    expert_partitioning = nnx.with_partitioning(
      nnx.initializers.lecun_normal(), ("expert", None, None)
    )
    key = rngs.params()
    k1, k2, k3 = jax.random.split(key, 3)
    self.gate = nnx.Param(expert_partitioning(k1, (n_experts, din, dhid)))
    self.up = nnx.Param(expert_partitioning(k2, (n_experts, din, dhid)))
    self.down = nnx.Param(expert_partitioning(k3, (n_experts, dhid, din)))

    self.shared_gate = tp_linear(din, dhid, ("fsdp", "tp"), rngs=rngs)
    self.shared_up = tp_linear(din, dhid, ("fsdp", "tp"), rngs=rngs)
    self.shared_down = tp_linear(dhid, din, ("tp", "fsdp"), rngs=rngs)

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
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

    shared = self.shared_down(
      jax.nn.silu(self.shared_gate(flat)) * self.shared_up(flat)
    )
    combined = combined + shared

    density = jnp.mean(probs, axis=0)
    chosen_mask = jax.nn.one_hot(top_idx, self.n_experts).sum(axis=1)
    chosen_frac = jnp.mean(chosen_mask, axis=0)
    aux_loss = self.n_experts * jnp.sum(density * chosen_frac)

    return combined.reshape(b, s, d), aux_loss


def gated_delta_net(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  alpha: jax.Array,
  beta: jax.Array,
) -> jax.Array:
  b, _, h, d = q.shape

  def step(state, inputs):
    q_t, k_t, v_t, a_t, beta_t = inputs
    predicted = jnp.einsum("bhde,bhe->bhd", state, k_t)
    delta = beta_t[..., None, None] * jnp.einsum(
      "bhd,bhe->bhde", v_t - predicted, k_t
    )
    state = a_t[..., None, None] * state + delta
    out = jnp.einsum("bhde,bhe->bhd", state, q_t)
    return state, out

  init_state = jnp.zeros((b, h, d, d))
  xs = (
    jnp.moveaxis(q, 1, 0),
    jnp.moveaxis(k, 1, 0),
    jnp.moveaxis(v, 1, 0),
    jnp.moveaxis(alpha, 1, 0),
    jnp.moveaxis(beta, 1, 0),
  )
  _, outs = jax.lax.scan(step, init_state, xs)
  return jnp.moveaxis(outs, 0, 1)


class GatedDeltaNetLayer(nnx.Module):
  def __init__(self, din, n_heads, head_dim, *, rngs):
    self.n_heads = n_heads
    self.head_dim = head_dim
    self.q_proj = tp_linear(
      din, n_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.k_proj = tp_linear(
      din, n_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.v_proj = tp_linear(
      din, n_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.gate_proj = nnx.Linear(din, 2 * n_heads, rngs=rngs)
    self.o_proj = tp_linear(
      n_heads * head_dim, din, ("tp", "fsdp"), rngs=rngs
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    b, s, _ = x.shape
    q = self.q_proj(x).reshape(b, s, self.n_heads, self.head_dim)
    k = self.k_proj(x).reshape(b, s, self.n_heads, self.head_dim)
    v = self.v_proj(x).reshape(b, s, self.n_heads, self.head_dim)
    q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    gates = jax.nn.sigmoid(self.gate_proj(x))
    alpha, beta = gates[..., : self.n_heads], gates[..., self.n_heads :]

    out = gated_delta_net(q, k, v, alpha, beta)
    return self.o_proj(out.reshape(b, s, self.n_heads * self.head_dim))


class Qwen3NextAttention(nnx.Module):
  def __init__(self, din, n_heads, n_kv_heads, head_dim, *, rngs):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.n_rep = n_heads // n_kv_heads
    self.rotary_dim = max(2, head_dim // 4 // 2 * 2)
    self.q_proj = tp_linear(
      din, n_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.k_proj = tp_linear(
      din, n_kv_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.v_proj = tp_linear(
      din, n_kv_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.gate_proj = nnx.Linear(din, n_heads * head_dim, rngs=rngs)
    self.o_proj = tp_linear(
      n_heads * head_dim, din, ("tp", "fsdp"), rngs=rngs
    )
    self.q_norm = RMSNorm(head_dim, rngs=rngs)
    self.k_norm = RMSNorm(head_dim, rngs=rngs)
    self.inv_freq = nnx.Variable(rope_freqs(self.rotary_dim))

  def __call__(
    self, x: jax.Array, positions: jax.Array, mask: jax.Array
  ) -> jax.Array:
    b, s, _ = x.shape
    q = self.q_proj(x).reshape(b, s, self.n_heads, self.head_dim)
    k = self.k_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim)
    v = self.v_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim)

    q = self.q_norm(q)
    k = self.k_norm(k)
    q = apply_partial_rope(q, positions, self.inv_freq.value, self.rotary_dim)
    k = apply_partial_rope(k, positions, self.inv_freq.value, self.rotary_dim)
    k = repeat_kv(k, self.n_rep)
    v = repeat_kv(v, self.n_rep)

    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(self.head_dim)
    scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(
      b, s, self.n_heads * self.head_dim
    )

    gate = jax.nn.sigmoid(self.gate_proj(x))
    return self.o_proj(gate * out)


class Qwen3NextBlock(nnx.Module):
  def __init__(  # noqa: PLR0913
    self,
    din,
    n_heads,
    n_kv_heads,
    head_dim,
    dhid,
    n_experts,
    n_active,
    *,
    is_linear,
    rngs,
  ):
    self.is_linear = is_linear
    self.attn_norm = RMSNorm(din, rngs=rngs)
    self.attn = (
      GatedDeltaNetLayer(din, n_heads, head_dim, rngs=rngs)
      if is_linear
      else Qwen3NextAttention(din, n_heads, n_kv_heads, head_dim, rngs=rngs)
    )
    self.ffn_norm = RMSNorm(din, rngs=rngs)
    self.ffn = SparseMoEFFN(din, dhid, n_experts, n_active, rngs=rngs)

  def __call__(
    self, x: jax.Array, positions: jax.Array, mask: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    normed = self.attn_norm(x)
    attn_out = (
      self.attn(normed) if self.is_linear else self.attn(normed, positions, mask)
    )
    x = x + attn_out
    ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
    return x + ffn_out, aux_loss


class Qwen3Next(nnx.Module):
  """Qwen3Next-style LM."""

  def __init__(  # noqa: PLR0913
    self,
    vocab_size,
    din,
    n_layers,
    n_heads,
    n_kv_heads,
    head_dim,
    dhid,
    n_experts,
    n_active,
    *,
    linear_every=4,
    aux_loss_coef=0.01,
    rngs,
  ):
    self.aux_loss_coef = aux_loss_coef
    self.embed = nnx.Embed(
      vocab_size,
      din,
      embedding_init=nnx.with_partitioning(
        nnx.initializers.normal(), ("fsdp", None)
      ),
      rngs=rngs,
    )
    self.blocks = tuple(
      Qwen3NextBlock(
        din,
        n_heads,
        n_kv_heads,
        head_dim,
        dhid,
        n_experts,
        n_active,
        is_linear=((i + 1) % linear_every != 0),
        rngs=rngs,
      )
      for i in range(n_layers)
    )
    self.final_norm = RMSNorm(din, rngs=rngs)
    self.lm_head = nnx.Linear(
      din,
      vocab_size,
      use_bias=False,
      kernel_init=nnx.with_partitioning(
        nnx.initializers.lecun_normal(), ("fsdp", None)
      ),
      rngs=rngs,
    )

  def __call__(
    self, token_ids: jax.Array, positions: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    seq_len = token_ids.shape[1]
    i = jnp.arange(seq_len)[:, None]
    j = jnp.arange(seq_len)[None, :]
    mask = j <= i

    hidden = self.embed(token_ids)
    total_aux_loss = jnp.array(0.0)
    for block in self.blocks:
      hidden, aux_loss = block(hidden, positions, mask)
      total_aux_loss = total_aux_loss + aux_loss
    hidden = self.final_norm(hidden)
    logits = self.lm_head(hidden)
    return logits, self.aux_loss_coef * total_aux_loss
