"""DeepSeek-V2-style Multi-head Latent Attention (MLA): K/V are jointly
compressed into a low-rank latent (much smaller than uncompressed KV
width) and decompressed per-head at attention time. RoPE cannot be
applied to the compressed latent directly -- rotating then compressing
is not equivalent to compressing then rotating, which breaks RoPE's
relative-position dot-product identity -- so positional information is
carried by a separate, small "decoupled RoPE" projection computed
directly from the uncompressed input and concatenated onto the
compressed "content" (no-positional-encoding, "nope") part before the
attention dot product. Verified live (see the plan doc): the decoupled
split produces a numerically different, causally-correct result;
skipping it (naively RoPE-rotating the shared latent) diverges and is
wrong. Value vectors carry no positional component and are sized
head_dim_nope only, not head_dim_nope + head_dim_rope.

This module does not implement the KV-cache memory savings MLA exists
to provide in production -- this repo's generate() is
full-prefix-recompute for every model in this suite (see the design
doc's Out of Scope section). MLA is architecturally faithful here, but
nothing exploits its smaller cache.
"""

import jax
from flax import nnx
from jax import numpy as jnp

from layers import RMSNorm, apply_rope, rope_freqs, tp_linear


class MLAAttention(nnx.Module):
  """Multi-head Latent Attention with decoupled RoPE."""

  def __init__(
    self, d_model, n_heads, d_latent, head_dim_nope, head_dim_rope, *, rngs
  ):
    """Construct an MLA attention block.

    Args:
      d_model: model (residual stream) dimensionality.
      n_heads: number of attention heads (MLA has no separate kv-head
        count -- all heads share one compressed KV latent, which is the
        whole point of the compression, replacing GQA's coarser
        kv-head-sharing with a much more aggressive shared latent).
      d_latent: compressed latent dimensionality, shared across all
        heads. Should be substantially smaller than
        n_heads * head_dim_nope (the real DeepSeek-V2 point) -- guidance:
        roughly (n_heads * head_dim_nope) / 4.
      head_dim_nope: per-head "content" (no positional encoding)
        dimensionality, decompressed from the shared latent.
      head_dim_rope: per-head decoupled-RoPE dimensionality, computed
        directly from the uncompressed input, not from the latent.
      rngs: random keys.
    """
    self.n_heads = n_heads
    self.head_dim_nope = head_dim_nope
    self.head_dim_rope = head_dim_rope

    # KV path: compress (replicated, small bottleneck) -> norm ->
    # decompress per-head (TP-sharded by head).
    self.down_kv = nnx.Linear(d_model, d_latent, use_bias=False, rngs=rngs)
    self.norm_kv = RMSNorm(d_latent, rngs=rngs)
    self.up_k = tp_linear(
      d_latent, n_heads * head_dim_nope, ("fsdp", "tp"), rngs=rngs
    )
    self.up_v = tp_linear(
      d_latent, n_heads * head_dim_nope, ("fsdp", "tp"), rngs=rngs
    )
    self.rope_k = tp_linear(
      d_model, n_heads * head_dim_rope, ("fsdp", "tp"), rngs=rngs
    )

    # Query path: same compress/decompress + decoupled-RoPE split.
    self.down_q = nnx.Linear(d_model, d_latent, use_bias=False, rngs=rngs)
    self.norm_q = RMSNorm(d_latent, rngs=rngs)
    self.up_q = tp_linear(
      d_latent, n_heads * head_dim_nope, ("fsdp", "tp"), rngs=rngs
    )
    self.rope_q = tp_linear(
      d_model, n_heads * head_dim_rope, ("fsdp", "tp"), rngs=rngs
    )

    # Output projection: value vectors carry head_dim_nope only (no
    # positional component), so this is sized from n_heads * head_dim_nope.
    self.o_proj = tp_linear(
      n_heads * head_dim_nope, d_model, ("tp", "fsdp"), rngs=rngs
    )
    # nnx.Variable wrap required -- see the identical note on
    # GQAAttention.inv_freq in layers.py (Task 1): bare jax.Array module
    # attributes are rejected by nnx.split/nnx.grad's graph flattening,
    # and a plain nnx.Variable (not nnx.Param) is correctly excluded
    # from the gradient pytree entirely.
    self.inv_freq = nnx.Variable(rope_freqs(head_dim_rope))

  def __call__(
    self, x: jax.Array, positions: jax.Array, mask: jax.Array
  ) -> jax.Array:
    """Apply multi-head latent attention.

    Args:
      x: input array, shape (batch, seq, d_model).
      positions: integer position ids, shape (batch, seq).
      mask: bool attention mask, shape (seq, seq), True = attend. MLA is
        always used with full causal masking in this suite (no local
        windowing -- that's Gemma-specific).

    Returns:
      jax.Array, same shape as x.
    """
    b, s, _ = x.shape

    latent_kv = self.norm_kv(self.down_kv(x))
    k_nope = self.up_k(latent_kv).reshape(b, s, self.n_heads, self.head_dim_nope)
    v = self.up_v(latent_kv).reshape(b, s, self.n_heads, self.head_dim_nope)

    k_rope = self.rope_k(x).reshape(b, s, self.n_heads, self.head_dim_rope)
    k_rope = apply_rope(k_rope, positions, self.inv_freq.value)
    k = jnp.concatenate([k_nope, k_rope], axis=-1)

    latent_q = self.norm_q(self.down_q(x))
    q_nope = self.up_q(latent_q).reshape(b, s, self.n_heads, self.head_dim_nope)
    q_rope = self.rope_q(x).reshape(b, s, self.n_heads, self.head_dim_rope)
    q_rope = apply_rope(q_rope, positions, self.inv_freq.value)
    q = jnp.concatenate([q_nope, q_rope], axis=-1)

    head_dim = self.head_dim_nope + self.head_dim_rope
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(head_dim)
    scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(
      b, s, self.n_heads * self.head_dim_nope
    )
    return self.o_proj(out)
