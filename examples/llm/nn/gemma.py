"""Gemma4-style LM.

Architecture design:
  - Attention: grouped-query attention (GQA), n_kv_heads query groups
    broadcast to n_heads via repeat_kv, softmax over head_dim ** -0.5
    scaled scores. Queries and keys are RMS-normed per head before RoPE
    ("QK-norm").
  - Attention pattern: interleaved local/global layers. Every
    global_every-th layer (1-indexed, default 4) is "global"; the rest
    are "local".
    - local layers: sliding-window causal attention (last local_window
      positions), full RoPE (rotary_fraction=1.0) at base theta=10k.
    - global layers: full-causal attention, dual-frequency partial RoPE
      ("p-RoPE": only the leading 25% of head_dim rotated) at base
      theta=1M.
  - KV-cache trick: key/value reuse on global layers -- the value
    projection is dropped and the (pre-rotation) key projection is reused
    as values (share_kv), shrinking the global-layer KV cache.
  - Normalization: RMSNorm (no mean-centering, no bias), pre-norm, with
    a separate norm before attention and FFN plus a final norm before the
    head.
  - FFN: dense GeGLU (gated GELU, column-parallel gate/up, row-parallel
    down).
  - Embeddings: untied -- a separate input nnx.Embed and output lm_head.
  - Residual structure: standard x = x + sublayer(norm(x)).
  - Sharding: 2D FSDP+TP -- column-parallel q/k/v/gate/up, row-parallel
    o/down projections; embedding and lm_head kernels FSDP-sharded on the
    vocab axis.

  Faithful to the real model:
  - interleaved local/global attention
  - dual-frequency p-RoPE
  - key-as-value reuse on global layers
  - QK-norm

  Divergences from the real model (real / implemented):
  - sandwich normalization (post-attn/post-FFN norms) / pre-norm only
  - sqrt(din) embedding scaling / unscaled embeddings
  - tied input/output embeddings / untied (separate lm_head)
  - per-layer embeddings (E2B/E4B) / one shared embedding table
  - cross-layer KV-cache sharing / each layer keeps its own KV
  - illustrative layer sizes, not any real variant's config
"""

import jax
from flax import nnx
from jax import numpy as jnp
from nn.layers import (
  GeGLU,
  RMSNorm,
  apply_partial_rope,
  make_causal_mask,
  repeat_kv,
  rope_freqs,
  tp_linear,
)


class GemmaAttention(nnx.Module):
  """Grouped-query attention."""

  def __init__(
    self,
    din,
    n_heads,
    n_kv_heads,
    head_dim,
    *,
    theta,
    rotary_fraction,
    share_kv,
    rngs,
  ):
    """Construct a Gemma4 attention block.

    Args:
      din: model (residual stream) dimensionality.
      n_heads: number of query heads.
      n_kv_heads: number of key/value heads.
      head_dim: dimensionality of each attention head.
      theta: RoPE base frequency for this layer (1M on global layers,
        10k on local layers).
      rotary_fraction: fraction of head_dim rotated by RoPE (0.25 on
        global layers, 1.0 -- full head -- on local layers).
      share_kv: if True, skip the value projection and reuse the
        (pre-rotation) key projection as values (global layers only).
      rngs: random keys.
    """
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.n_rep = n_heads // n_kv_heads
    self.share_kv = share_kv
    self.rotary_dim = max(2, int(rotary_fraction * head_dim) // 2 * 2)
    self.q_proj = tp_linear(
      din, n_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    self.k_proj = tp_linear(
      din, n_kv_heads * head_dim, ("fsdp", "tp"), rngs=rngs
    )
    if not share_kv:
      self.v_proj = tp_linear(
        din, n_kv_heads * head_dim, ("fsdp", "tp"), rngs=rngs
      )
    self.o_proj = tp_linear(
      n_heads * head_dim, din, ("tp", "fsdp"), rngs=rngs
    )
    self.q_norm = RMSNorm(head_dim, rngs=rngs)
    self.k_norm = RMSNorm(head_dim, rngs=rngs)
    self.inv_freq = nnx.Variable(rope_freqs(self.rotary_dim, theta=theta))

  def __call__(
    self, x: jax.Array, positions: jax.Array, mask: jax.Array
  ) -> jax.Array:
    """Apply grouped-query self-attention.

    Args:
      x: input array, shape (batch, seq, din).
      positions: integer position ids, shape (batch, seq).
      mask: bool attention mask, shape (seq, seq), True = attend, from
        make_causal_mask.

    Returns:
      jax.Array, same shape as x.
    """
    b, s, _ = x.shape
    q = self.q_proj(x).reshape(b, s, self.n_heads, self.head_dim)
    k_content = self.k_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim)
    v = (
      k_content
      if self.share_kv
      else self.v_proj(x).reshape(b, s, self.n_kv_heads, self.head_dim)
    )

    q = self.q_norm(q)
    q = apply_partial_rope(q, positions, self.inv_freq.value, self.rotary_dim)
    k = apply_partial_rope(
      self.k_norm(k_content), positions, self.inv_freq.value, self.rotary_dim
    )
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


class GemmaTransformerBlock(nnx.Module):
  """Pre-norm transformer block."""

  def __init__(  # noqa: PLR0913
    self, din, n_heads, n_kv_heads, head_dim, dhid, *, is_global, rngs
  ):
    """Construct a Gemma transformer block.

    Args:
      din: model (residual stream) dimensionality.
      n_heads: number of query heads.
      n_kv_heads: number of key/value heads.
      head_dim: dimensionality of each attention head.
      dhid: feed-forward hidden dimensionality.
      is_global: whether this is a "global" (full-causal, p-RoPE,
        shared-kv) layer or a "local" (sliding-window, full-RoPE) one.
      rngs: random keys.
    """
    self.is_global = is_global
    self.attn_norm = RMSNorm(din, rngs=rngs)
    self.attn = GemmaAttention(
      din,
      n_heads,
      n_kv_heads,
      head_dim,
      theta=1_000_000.0 if is_global else 10_000.0,
      rotary_fraction=0.25 if is_global else 1.0,
      share_kv=is_global,
      rngs=rngs,
    )
    self.ffn_norm = RMSNorm(din, rngs=rngs)
    self.ffn = GeGLU(din, dhid, rngs=rngs)

  def __call__(
    self, x: jax.Array, positions: jax.Array, mask: jax.Array
  ) -> jax.Array:
    """Apply the block.

    Args:
      x: input array, shape (batch, seq, din).
      positions: integer position ids, shape (batch, seq).
      mask: bool attention mask, shape (seq, seq).

    Returns:
      jax.Array, same shape as x.
    """
    x = x + self.attn(self.attn_norm(x), positions, mask)
    x = x + self.ffn(self.ffn_norm(x))
    return x


class Gemma4(nnx.Module):
  """Gemma4-style LM."""

  def __init__(  # noqa: PLR0913
    self,
    vocab_size,
    din,
    n_layers,
    n_heads,
    n_kv_heads,
    head_dim,
    dhid,
    local_window,
    *,
    global_every=4,
    rngs,
  ):
    """Construct a GemmaLLM.

    Args:
      vocab_size: token vocabulary size.
      din: model (residual stream) dimensionality.
      n_layers: number of transformer blocks.
      n_heads: number of query heads.
      n_kv_heads: number of key/value heads.
      head_dim: dimensionality of each attention head.
      dhid: feed-forward hidden dimensionality.
      local_window: sliding-window size for "local" attention layers.
      global_every: every global_every-th layer (1-indexed) is a full
        causal ("global") attention layer; the rest are local.
      rngs: random keys.
    """
    self.local_window = local_window
    self.embed = nnx.Embed(
      vocab_size,
      din,
      embedding_init=nnx.with_partitioning(
        nnx.initializers.normal(), ("fsdp", None)
      ),
      rngs=rngs,
    )
    self.blocks = tuple(
      GemmaTransformerBlock(
        din,
        n_heads,
        n_kv_heads,
        head_dim,
        dhid,
        is_global=((i + 1) % global_every == 0),
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
    """Compute next-token logits for a batch of token sequences.

    Args:
      token_ids: integer token ids, shape (batch, seq).
      positions: integer position ids, shape (batch, seq).

    Returns:
      a tuple of logits, aux_loss
    """
    seq_len = token_ids.shape[1]
    global_mask = make_causal_mask(seq_len)
    local_mask = make_causal_mask(seq_len, window=self.local_window)

    hidden = self.embed(token_ids)
    for block in self.blocks:
      mask = global_mask if block.is_global else local_mask
      hidden = block(hidden, positions, mask)

    hidden = self.final_norm(hidden)
    logits = self.lm_head(hidden)
    return logits, jnp.array(0.0)
