"""Qwen3-Next-style decoder-only transformer: linear-attention/
Transformer hybrid + sparse MoE FFN.

75% of layers are Gated DeltaNet -- a linear (constant-memory-per-step)
recurrent attention computed via the delta rule, no softmax, no
quadratic seq x seq score matrix -- and the remaining 25% are standard
GQA attention with partial RoPE (leading 25% of the head) and an output
gate. This is the axis none of this suite's other models cover: every
other model here is quadratic self-attention throughout; Qwen3-Next
interleaves it with a genuinely sub-quadratic sequence mixer.

The Gated DeltaNet recurrence is implemented as a plain `jax.lax.scan`
over time steps, which is O(seq_len) sequential steps -- correct and
easy to follow, but not the real chunked/parallel-form kernel used in
production Qwen3-Next; fine at this suite's toy sequence lengths.
Multi-token prediction (a real Qwen3-Next feature) is out of scope
here -- see DeepSeekV4Dense for this suite's other new-attention-
mechanism reference instead.
"""

import jax
from flax import nnx
from jax import numpy as jnp
from layers import RMSNorm, apply_partial_rope, repeat_kv, rope_freqs, tp_linear


class SparseMoEFFN(nnx.Module):
  """Top-k-routed mixture-of-experts feed-forward block with real
  capacity-based dispatch/combine (not dense-compute-then-select).
  Expert weights are stacked into single tensors with a leading
  n_experts axis so that axis can be sharded across a mesh's "expert"
  axis.
  """

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
    self.gate = nnx.Param(expert_partitioning(k1, (n_experts, d_model, d_ff)))
    self.up = nnx.Param(expert_partitioning(k2, (n_experts, d_model, d_ff)))
    self.down = nnx.Param(expert_partitioning(k3, (n_experts, d_ff, d_model)))

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


def gated_delta_net(
  q: jax.Array,
  k: jax.Array,
  v: jax.Array,
  alpha: jax.Array,
  beta: jax.Array,
) -> jax.Array:
  """Sequential Gated DeltaNet recurrence.

  Maintains a (head_dim x head_dim) associative-memory state per head,
  decayed each step by a data-dependent gate `alpha` and corrected
  toward the true value via the delta rule, scaled by a data-dependent
  write-rate `beta`: state_t = alpha_t * state_{t-1} + beta_t * (v_t -
  state_{t-1} @ k_t) (outer) k_t. Output is state_t @ q_t.

  Args:
    q: queries, shape (batch, seq, heads, head_dim).
    k: keys, shape (batch, seq, heads, head_dim).
    v: values, shape (batch, seq, heads, head_dim).
    alpha: decay gate in (0, 1), shape (batch, seq, heads).
    beta: write-rate gate in (0, 1), shape (batch, seq, heads).

  Returns:
    jax.Array, shape (batch, seq, heads, head_dim).
  """
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
  """Linear-attention sequence mixer via the (gated) delta rule."""

  def __init__(self, d_model, n_heads, head_dim, *, rngs):
    """Construct a Gated DeltaNet layer.

    Args:
      d_model: model (residual stream) dimensionality.
      n_heads: number of heads.
      head_dim: dimensionality of each head.
      rngs: random keys.
    """
    self.n_heads = n_heads
    self.head_dim = head_dim
    self.q_proj = tp_linear(
      d_model, n_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.k_proj = tp_linear(
      d_model, n_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.v_proj = tp_linear(
      d_model, n_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.gate_proj = nnx.Linear(d_model, 2 * n_heads, rngs=rngs)
    self.o_proj = tp_linear(
      n_heads * head_dim, d_model, ("tp", "fsdp"), rngs=rngs
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Apply the Gated DeltaNet layer.

    Args:
      x: input array, shape (batch, seq, d_model).

    Returns:
      jax.Array, same shape as x.
    """
    b, s, _ = x.shape
    q = self.q_proj(x).reshape(b, s, self.n_heads, self.head_dim)
    k = self.k_proj(x).reshape(b, s, self.n_heads, self.head_dim)
    v = self.v_proj(x).reshape(b, s, self.n_heads, self.head_dim)
    gates = jax.nn.sigmoid(self.gate_proj(x))
    alpha, beta = gates[..., : self.n_heads], gates[..., self.n_heads :]

    out = gated_delta_net(q, k, v, alpha, beta)
    return self.o_proj(out.reshape(b, s, self.n_heads * self.head_dim))


class Qwen3NextAttention(nnx.Module):
  """Standard GQA attention with partial RoPE and an output gate."""

  def __init__(self, d_model, n_heads, n_kv_heads, head_dim, *, rngs):
    """Construct a standard-attention layer.

    Args:
      d_model: model (residual stream) dimensionality.
      n_heads: number of query heads.
      n_kv_heads: number of key/value heads.
      head_dim: dimensionality of each attention head.
      rngs: random keys.
    """
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.n_rep = n_heads // n_kv_heads
    self.rotary_dim = max(2, head_dim // 4 // 2 * 2)
    self.q_proj = tp_linear(
      d_model, n_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.k_proj = tp_linear(
      d_model, n_kv_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.v_proj = tp_linear(
      d_model, n_kv_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.gate_proj = nnx.Linear(d_model, n_heads * head_dim, rngs=rngs)
    self.o_proj = tp_linear(
      n_heads * head_dim, d_model, ("tp", "fsdp"), rngs=rngs
    )
    self.inv_freq = nnx.Variable(rope_freqs(self.rotary_dim))

  def __call__(
    self, x: jax.Array, positions: jax.Array, mask: jax.Array
  ) -> jax.Array:
    """Apply gated grouped-query self-attention.

    Args:
      x: input array, shape (batch, seq, d_model).
      positions: integer position ids, shape (batch, seq).
      mask: bool attention mask, shape (seq, seq).

    Returns:
      jax.Array, same shape as x.
    """
    b, s, _ = x.shape
    q = self.q_proj(x).reshape(b, s, self.n_heads, self.head_dim)
    k = self.k_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim)
    v = self.v_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim)

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
  """Pre-norm transformer block: Gated DeltaNet or standard attention,
  plus a sparse MoE FFN.
  """

  def __init__(  # noqa: PLR0913
    self,
    d_model,
    n_heads,
    n_kv_heads,
    head_dim,
    d_ff,
    n_experts,
    n_active,
    *,
    is_linear,
    rngs,
  ):
    """Construct a Qwen3-Next transformer block.

    Args:
      d_model: model (residual stream) dimensionality.
      n_heads: number of attention/DeltaNet heads.
      n_kv_heads: number of key/value heads (standard-attention layers
        only).
      head_dim: dimensionality of each head.
      d_ff: feed-forward hidden dimensionality of each expert.
      n_experts: total experts.
      n_active: active experts per token (top-k).
      is_linear: whether this is a Gated DeltaNet layer (True, 75% of
        layers) or a standard-attention layer (False, 25%).
      rngs: random keys.
    """
    self.is_linear = is_linear
    self.attn_norm = RMSNorm(d_model, rngs=rngs)
    self.attn = (
      GatedDeltaNetLayer(d_model, n_heads, head_dim, rngs=rngs)
      if is_linear
      else Qwen3NextAttention(d_model, n_heads, n_kv_heads, head_dim, rngs=rngs)
    )
    self.ffn_norm = RMSNorm(d_model, rngs=rngs)
    self.ffn = SparseMoEFFN(d_model, d_ff, n_experts, n_active, rngs=rngs)

  def __call__(
    self, x: jax.Array, positions: jax.Array, mask: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """Apply the block.

    Args:
      x: input array, shape (batch, seq, d_model).
      positions: integer position ids, shape (batch, seq).
      mask: bool attention mask, shape (seq, seq) (standard-attention
        layers only -- unused on Gated DeltaNet layers).

    Returns:
      a tuple (output, aux_loss): output has the same shape as x.
    """
    normed = self.attn_norm(x)
    attn_out = (
      self.attn(normed) if self.is_linear else self.attn(normed, positions, mask)
    )
    x = x + attn_out
    ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
    return x + ffn_out, aux_loss


class Qwen3NextLLM(nnx.Module):
  """Decoder-only transformer in the Qwen3-Next architectural family:
  linear-attention (Gated DeltaNet) / standard-attention hybrid with a
  sparse MoE FFN.
  """

  def __init__(  # noqa: PLR0913
    self,
    vocab_size,
    d_model,
    n_layers,
    n_heads,
    n_kv_heads,
    head_dim,
    d_ff,
    n_experts,
    n_active,
    *,
    linear_every=4,
    aux_loss_coef=0.01,
    rngs,
  ):
    """Construct a Qwen3NextLLM.

    Args:
      vocab_size: token vocabulary size.
      d_model: model (residual stream) dimensionality.
      n_layers: number of transformer blocks.
      n_heads: number of attention/DeltaNet heads.
      n_kv_heads: number of key/value heads (standard-attention layers).
      head_dim: dimensionality of each head.
      d_ff: feed-forward hidden dimensionality of each expert.
      n_experts: total experts per block.
      n_active: active experts per token (top-k) per block.
      linear_every: every linear_every-th layer (1-indexed) is a
        standard-attention layer; the rest are Gated DeltaNet (3:1
        ratio at the default of 4, matching real Qwen3-Next).
      aux_loss_coef: weight applied to each block's load-balancing
        aux_loss before summing across blocks.
      rngs: random keys.
    """
    self.aux_loss_coef = aux_loss_coef
    self.embed = nnx.Embed(
      vocab_size,
      d_model,
      embedding_init=nnx.with_partitioning(
        nnx.initializers.normal(), ("fsdp", None)
      ),
      rngs=rngs,
    )
    self.blocks = tuple(
      Qwen3NextBlock(
        d_model,
        n_heads,
        n_kv_heads,
        head_dim,
        d_ff,
        n_experts,
        n_active,
        is_linear=((i + 1) % linear_every != 0),
        rngs=rngs,
      )
      for i in range(n_layers)
    )
    self.final_norm = RMSNorm(d_model, rngs=rngs)
    self.lm_head = nnx.Linear(
      d_model,
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
    """Compute next-token logits for a batch of token sequences.

    Args:
      token_ids: integer token ids, shape (batch, seq).
      positions: integer position ids, shape (batch, seq).

    Returns:
      a tuple (logits, aux_loss): logits has shape
      (batch, seq, vocab_size); aux_loss is aux_loss_coef times the
      summed per-block load-balancing loss (standard-attention layers'
      MoE FFNs only -- Gated DeltaNet layers still route through a
      SparseMoEFFN, same as standard-attention layers).
    """
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


def Qwen3NextHybrid(vocab_size, **kwargs):
  return Qwen3NextLLM(vocab_size, **kwargs)
