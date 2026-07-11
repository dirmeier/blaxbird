"""Shared primitives for the llm_reference example suite (Gemma-4-style,
DeepSeek-MLA-style, Mixtral-style decoder-only transformers).

TP-sharded projections use nnx.with_partitioning on their kernel_init so
blaxbird.train_fn's mesh= argument can shard them via
nnx.get_named_sharding -- unannotated parameters (e.g. RMSNorm weights)
default to fully replicated. No mesh is threaded into any __call__: all
sharding here is construction-time weight annotation only, matching this
repo's existing sharding idiom (see examples/fsdp_tp_demo).
"""

import jax
from flax import nnx
from jax import numpy as jnp


def rope_freqs(head_dim: int, theta: float = 10_000.0) -> jax.Array:
  """Compute RoPE inverse frequencies.

  Args:
    head_dim: dimensionality to rotate (must be even).
    theta: RoPE base frequency.

  Returns:
    jax.Array, shape (head_dim // 2,).
  """
  return 1.0 / (theta ** (jnp.arange(0, head_dim, 2) / head_dim))


def apply_rope(
  x: jax.Array, positions: jax.Array, inv_freq: jax.Array
) -> jax.Array:
  """Apply rotary position embeddings.

  Args:
    x: input array, shape (batch, seq, n_heads, dim) where dim ==
      2 * inv_freq.shape[0].
    positions: integer position ids, shape (batch, seq).
    inv_freq: RoPE inverse frequencies from rope_freqs.

  Returns:
    jax.Array, same shape as x.
  """
  freqs = positions[:, :, None] * inv_freq[None, None, :]
  cos = jnp.cos(freqs)[:, :, None, :]
  sin = jnp.sin(freqs)[:, :, None, :]
  x1, x2 = jnp.split(x, 2, axis=-1)
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
  """Broadcast grouped-query-attention key/value heads to n_heads.

  Args:
    x: input array, shape (batch, seq, n_kv_heads, head_dim).
    n_rep: number of query heads sharing each kv head
      (n_heads // n_kv_heads).

  Returns:
    jax.Array, shape (batch, seq, n_kv_heads * n_rep, head_dim).
  """
  if n_rep == 1:
    return x
  b, s, kvh, hd = x.shape
  x = jnp.broadcast_to(x[:, :, :, None, :], (b, s, kvh, n_rep, hd))
  return x.reshape(b, s, kvh * n_rep, hd)


def make_causal_mask(seq_len: int, window: int | None = None) -> jax.Array:
  """Build a causal (optionally sliding-window / "local") attention mask.

  Args:
    seq_len: sequence length.
    window: if given, restrict attention to the last `window` positions
      (Gemma-style "local" attention layers); if None, full causal
      ("global") attention -- always the case for DeepSeek and Mixtral in
      this suite.

  Returns:
    bool jax.Array, shape (seq_len, seq_len), True where attention is
    allowed (query position i may attend to key position j).
  """
  i = jnp.arange(seq_len)[:, None]
  j = jnp.arange(seq_len)[None, :]
  causal = j <= i
  if window is not None:
    causal = causal & (j > i - window)
  return causal


class RMSNorm(nnx.Module):
  """Root-mean-square layer normalization (no mean-centering, no bias)."""

  def __init__(self, dim, *, rngs, eps=1e-6):
    """Construct an RMSNorm layer.

    Args:
      dim: feature dimensionality.
      rngs: random keys (unused -- weight is initialized to ones -- kept
        for interface consistency with every other block in this suite).
      eps: numerical-stability constant.
    """
    del rngs
    self.weight = nnx.Param(jnp.ones((dim,)))
    self.eps = eps

  def __call__(self, x: jax.Array) -> jax.Array:
    """Normalize the last axis of x by its RMS, then scale.

    Args:
      x: input array, shape (..., dim).

    Returns:
      jax.Array, same shape as x.
    """
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(var + self.eps)
    return x * self.weight.value


def tp_linear(d_in, d_out, partition_spec, *, rngs, use_bias=False):
  """Construct a nnx.Linear whose kernel carries a sharding annotation.

  Args:
    d_in: input feature dimensionality.
    d_out: output feature dimensionality.
    partition_spec: a 2-tuple of mesh-axis names (or None) passed to
      nnx.with_partitioning on the kernel initializer -- e.g.
      ("fsdp", "tp") for column-parallel, ("tp", "fsdp") for row-parallel.
      Resolved to a real per-device shard only when the returned module's
      state is passed through nnx.get_named_sharding(state, mesh) inside
      blaxbird.train_fn; constructing this module standalone (no mesh) is
      unaffected -- same pattern as examples/fsdp_tp_demo's ShardedMLP.
    rngs: random keys.
    use_bias: whether to include a bias term. Every projection in this
      suite uses use_bias=False, matching the real
      Gemma/DeepSeek/Mixtral architectures.

  Returns:
    a nnx.Linear with a with_partitioning-annotated kernel_init.
  """
  return nnx.Linear(
    d_in,
    d_out,
    use_bias=use_bias,
    kernel_init=nnx.with_partitioning(
      nnx.initializers.lecun_normal(), partition_spec
    ),
    rngs=rngs,
  )


class GQAAttention(nnx.Module):
  """Grouped-query attention with RoPE, TP-sharded (column-parallel qkv,
  row-parallel output projection -- standard Megatron-style tensor
  parallelism)."""

  def __init__(self, d_model, n_heads, n_kv_heads, head_dim, *, rngs):
    """Construct a GQA attention block.

    Args:
      d_model: model (residual stream) dimensionality.
      n_heads: number of query heads.
      n_kv_heads: number of key/value heads (n_heads must be a multiple
        of n_kv_heads; n_kv_heads == n_heads recovers standard MHA,
        n_kv_heads == 1 recovers MQA).
      head_dim: dimensionality of each attention head.
      rngs: random keys.
    """
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.n_rep = n_heads // n_kv_heads
    self.q_proj = tp_linear(
      d_model, n_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.k_proj = tp_linear(
      d_model, n_kv_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.v_proj = tp_linear(
      d_model, n_kv_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.o_proj = tp_linear(
      n_heads * head_dim, d_model, ("tp", "fsdp"), rngs=rngs
    )
    self.inv_freq = nnx.Variable(rope_freqs(head_dim))

  def __call__(
    self, x: jax.Array, positions: jax.Array, mask: jax.Array
  ) -> jax.Array:
    """Apply grouped-query self-attention.

    Args:
      x: input array, shape (batch, seq, d_model).
      positions: integer position ids, shape (batch, seq).
      mask: bool attention mask, shape (seq, seq), True = attend, from
        make_causal_mask.

    Returns:
      jax.Array, same shape as x.
    """
    b, s, _ = x.shape
    q = self.q_proj(x).reshape(b, s, self.n_heads, self.head_dim)
    k = self.k_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim)
    v = self.v_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim)

    q = apply_rope(q, positions, self.inv_freq.value)
    k = apply_rope(k, positions, self.inv_freq.value)
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
    return self.o_proj(out)


class GeGLU(nnx.Module):
  """Gated GELU MLP, TP-sharded (column-parallel gate/up, row-parallel
  down -- same pattern as GQAAttention's projections)."""

  def __init__(self, d_model, d_ff, *, rngs):
    """Construct a GeGLU feed-forward block.

    Args:
      d_model: model (residual stream) dimensionality.
      d_ff: hidden (expansion) dimensionality.
      rngs: random keys.
    """
    self.gate = tp_linear(d_model, d_ff, ("fsdp", "tp"), rngs=rngs)
    self.up = tp_linear(d_model, d_ff, ("fsdp", "tp"), rngs=rngs)
    self.down = tp_linear(d_ff, d_model, ("tp", "fsdp"), rngs=rngs)

  def __call__(self, x: jax.Array) -> jax.Array:
    """Apply the GeGLU transform.

    Args:
      x: input array, shape (..., d_model).

    Returns:
      jax.Array, same shape as x.
    """
    return self.down(jax.nn.gelu(self.gate(x)) * self.up(x))
