"""Shared primitives for the llm_reference example suite (Gemma-4-
style, DeepSeek-V4-style, Qwen3-Next-style decoder-only transformers).

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


def apply_partial_rope(
  x: jax.Array, positions: jax.Array, inv_freq: jax.Array, rotary_dim: int
) -> jax.Array:
  """Apply RoPE to only the leading `rotary_dim` of the head axis,
  leaving the remainder unrotated (Gemma-4-style "p-RoPE" on global
  attention layers; Qwen3-Next-style partial RoPE on standard attention
  layers).

  Args:
    x: input array, shape (batch, seq, n_heads, dim).
    positions: integer position ids, shape (batch, seq).
    inv_freq: RoPE inverse frequencies, shape (rotary_dim // 2,).
    rotary_dim: number of leading dims to rotate (must be even, <= dim).

  Returns:
    jax.Array, same shape as x.
  """
  x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
  return jnp.concatenate(
    [apply_rope(x_rot, positions, inv_freq), x_pass], axis=-1
  )


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
      ("global") attention.

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
      Gemma/DeepSeek/Qwen3-Next architectures.

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


class GeGLU(nnx.Module):
  """Gated GELU MLP, TP-sharded (column-parallel gate/up, row-parallel
  down -- same pattern as this suite's attention projections).
  """

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
