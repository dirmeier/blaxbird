"""Gemma-4-style decoder-only transformer: GQA + RoPE + interleaved
local/global attention + dense GeGLU FFN, TP+FSDP-sharded.
"""

import jax
from flax import nnx
from jax import numpy as jnp

from layers import GQAAttention, GeGLU, RMSNorm, make_causal_mask


class GemmaTransformerBlock(nnx.Module):
  """Pre-norm transformer block: GQA attention + dense GeGLU FFN."""

  def __init__(self, d_model, n_heads, n_kv_heads, head_dim, d_ff, *, rngs):
    """Construct a Gemma transformer block.

    Args:
      d_model: model (residual stream) dimensionality.
      n_heads: number of query heads.
      n_kv_heads: number of key/value heads.
      head_dim: dimensionality of each attention head.
      d_ff: feed-forward hidden dimensionality.
      rngs: random keys.
    """
    self.attn_norm = RMSNorm(d_model, rngs=rngs)
    self.attn = GQAAttention(d_model, n_heads, n_kv_heads, head_dim, rngs=rngs)
    self.ffn_norm = RMSNorm(d_model, rngs=rngs)
    self.ffn = GeGLU(d_model, d_ff, rngs=rngs)

  def __call__(
    self, x: jax.Array, positions: jax.Array, mask: jax.Array
  ) -> jax.Array:
    """Apply the block.

    Args:
      x: input array, shape (batch, seq, d_model).
      positions: integer position ids, shape (batch, seq).
      mask: bool attention mask, shape (seq, seq).

    Returns:
      jax.Array, same shape as x.
    """
    x = x + self.attn(self.attn_norm(x), positions, mask)
    x = x + self.ffn(self.ffn_norm(x))
    return x


class GemmaLLM(nnx.Module):
  """Decoder-only transformer in the Gemma-4 architectural family.

  Interleaves "local" (sliding-window) and "global" (full causal)
  attention layers -- every `global_every`-th layer is global, the rest
  are local, matching Gemma 2/3/4's actual design choice. Dense FFN
  only (no MoE -- that's MixtralSMoE's role in this suite).
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
    local_window,
    *,
    global_every=4,
    rngs,
  ):
    """Construct a GemmaLLM.

    Args:
      vocab_size: token vocabulary size.
      d_model: model (residual stream) dimensionality.
      n_layers: number of transformer blocks.
      n_heads: number of query heads.
      n_kv_heads: number of key/value heads.
      head_dim: dimensionality of each attention head.
      d_ff: feed-forward hidden dimensionality.
      local_window: sliding-window size for "local" attention layers.
      global_every: every global_every-th layer (1-indexed) is a full
        causal ("global") attention layer; the rest are local.
      rngs: random keys.
    """
    self.local_window = local_window
    self.global_every = global_every
    self.embed = nnx.Embed(
      vocab_size,
      d_model,
      embedding_init=nnx.with_partitioning(
        nnx.initializers.normal(), ("fsdp", None)
      ),
      rngs=rngs,
    )
    self.blocks = tuple(
      GemmaTransformerBlock(d_model, n_heads, n_kv_heads, head_dim, d_ff, rngs=rngs)
      for _ in range(n_layers)
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
      (batch, seq, vocab_size); aux_loss is always jnp.array(0.0) (dense
      model, no MoE) -- kept for interface uniformity with MixtralSMoE so
      objective.py's causal_lm works unmodified across this suite.
    """
    seq_len = token_ids.shape[1]
    global_mask = make_causal_mask(seq_len)
    local_mask = make_causal_mask(seq_len, window=self.local_window)

    hidden = self.embed(token_ids)
    for i, block in enumerate(self.blocks):
      is_global = (i + 1) % self.global_every == 0
      mask = global_mask if is_global else local_mask
      hidden = block(hidden, positions, mask)

    hidden = self.final_norm(hidden)
    logits = self.lm_head(hidden)
    return logits, jnp.array(0.0)


def GemmaDense(vocab_size, **kwargs):
  return GemmaLLM(vocab_size, **kwargs)
