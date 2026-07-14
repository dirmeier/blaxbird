"""DeepSeek-V4-style decoder-only transformer: hybrid Compressed Sparse
Attention (CSA) + Heavily Compressed Attention (HCA) with learnable
attention sinks, RMSNorm, pre-norm, dense GeGLU FFN.

Real DeepSeek-V4 pairs this attention mechanism with MLA-style latent
KV compression (see DeepSeekMLA in this suite for that axis) and a
1.6T-parameter DeepSeekMoE FFN with shared experts (see MixtralSMoE for
the MoE-routing axis in this suite). This reference implementation
isolates just the new attention mechanism -- plain GQA projections,
dense FFN -- so each axis is demonstrated independently elsewhere in
this suite rather than combined into one file.

CSA pools keys/values into small blocks (light compression) and
attends to only the top-k most relevant blocks per query (real,
selective sparse attention -- not dense-then-mask). HCA pools into
larger blocks (heavy compression) and attends densely to all of them,
giving a cheap global view that complements CSA's sharper, selective
one. Both branches restrict attention to fully-completed blocks (mean-
pooled here, not the real softmax-gated pooling + FP4 lightning
indexer) -- the most recent < block_size tokens are only visible once
their block completes. Real V4 compensates with an explicit raw
sliding-window branch over recent tokens; omitted here since this
suite's toy sequence lengths (~16-32 tokens) would make that window
cover almost the whole sequence anyway.
"""

import jax
from flax import nnx
from jax import numpy as jnp
from layers import GeGLU, RMSNorm, apply_rope, repeat_kv, rope_freqs, tp_linear


def pool_kv_blocks(x: jax.Array, block_size: int) -> jax.Array:
  """Mean-pool (batch, seq, heads, dim) into non-overlapping blocks
  along seq, dropping any trailing partial block.

  Args:
    x: input array, shape (batch, seq, heads, dim).
    block_size: number of positions per block.

  Returns:
    jax.Array, shape (batch, seq // block_size, heads, dim).
  """
  b, s, h, d = x.shape
  n_blocks = s // block_size
  x = x[:, : n_blocks * block_size]
  return x.reshape(b, n_blocks, block_size, h, d).mean(axis=2)


def make_block_causal_mask(seq_len: int, block_size: int) -> jax.Array:
  """Build a causal mask from query positions to fully-completed blocks.

  Args:
    seq_len: sequence length (number of queries).
    block_size: number of positions per key/value block.

  Returns:
    bool jax.Array, shape (seq_len, seq_len // block_size), True where
    the block lies entirely at or before the query position.
  """
  n_blocks = seq_len // block_size
  query_pos = jnp.arange(seq_len)[:, None]
  block_last_pos = (jnp.arange(n_blocks) + 1) * block_size - 1
  return block_last_pos[None, :] <= query_pos


class DeepSeek4Attention(nnx.Module):
  """Hybrid CSA + HCA attention with learnable attention sinks."""

  def __init__(  # noqa: PLR0913
    self,
    d_model,
    n_heads,
    n_kv_heads,
    head_dim,
    *,
    csa_block_size,
    csa_top_k,
    hca_block_size,
    rngs,
  ):
    """Construct a DeepSeek-V4-style hybrid attention block.

    Args:
      d_model: model (residual stream) dimensionality.
      n_heads: number of query heads.
      n_kv_heads: number of key/value heads.
      head_dim: dimensionality of each attention head.
      csa_block_size: key/value pooling block size for the Compressed
        Sparse Attention branch (light compression, top-k selective).
      csa_top_k: number of CSA blocks attended to per query.
      hca_block_size: key/value pooling block size for the Heavily
        Compressed Attention branch (heavy compression, dense).
      rngs: random keys.
    """
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.n_rep = n_heads // n_kv_heads
    self.csa_block_size = csa_block_size
    self.csa_top_k = csa_top_k
    self.hca_block_size = hca_block_size
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
      2 * n_heads * head_dim, d_model, ("tp", "fsdp"), rngs=rngs
    )
    self.inv_freq = nnx.Variable(rope_freqs(head_dim))
    self.csa_sink = nnx.Param(jnp.zeros((n_heads,)))
    self.hca_sink = nnx.Param(jnp.zeros((n_heads,)))

  def _branch(self, q, k_blocks, v_blocks, block_mask, sink, top_k):
    """Compute one compressed-attention branch (CSA if top_k is given,
    HCA -- dense over all blocks -- otherwise), with a learnable
    attention sink competing for softmax probability mass so the real
    block weights can sum to less than one.
    """
    b, h, s, d = q.shape
    n_blocks = k_blocks.shape[2]
    scores = jnp.einsum("bhsd,bhnd->bhsn", q, k_blocks) / jnp.sqrt(d)
    scores = jnp.where(block_mask[None, None, :, :], scores, -jnp.inf)
    sink_logit = jnp.broadcast_to(sink[None, :, None, None], (b, h, s, 1))

    if top_k is not None and top_k < n_blocks:
      top_scores, top_idx = jax.lax.top_k(scores, top_k)
      combined = jnp.concatenate([top_scores, sink_logit], axis=-1)
      weights = jax.nn.softmax(combined, axis=-1)[..., :-1]
      v_full = jnp.broadcast_to(v_blocks[:, :, None], (b, h, s, n_blocks, d))
      v_sel = jnp.take_along_axis(v_full, top_idx[..., None], axis=3)
      return jnp.einsum("bhsk,bhskd->bhsd", weights, v_sel)

    combined = jnp.concatenate([scores, sink_logit], axis=-1)
    weights = jax.nn.softmax(combined, axis=-1)[..., :-1]
    return jnp.einsum("bhsn,bhnd->bhsd", weights, v_blocks)

  def __call__(self, x: jax.Array, positions: jax.Array) -> jax.Array:
    """Apply hybrid CSA+HCA self-attention.

    Args:
      x: input array, shape (batch, seq, d_model).
      positions: integer position ids, shape (batch, seq).

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

    csa_k = jnp.transpose(pool_kv_blocks(k, self.csa_block_size), (0, 2, 1, 3))
    csa_v = jnp.transpose(pool_kv_blocks(v, self.csa_block_size), (0, 2, 1, 3))
    hca_k = jnp.transpose(pool_kv_blocks(k, self.hca_block_size), (0, 2, 1, 3))
    hca_v = jnp.transpose(pool_kv_blocks(v, self.hca_block_size), (0, 2, 1, 3))
    q = jnp.transpose(q, (0, 2, 1, 3))

    csa_mask = make_block_causal_mask(s, self.csa_block_size)
    hca_mask = make_block_causal_mask(s, self.hca_block_size)

    csa_out = self._branch(
      q, csa_k, csa_v, csa_mask, self.csa_sink.value, self.csa_top_k
    )
    hca_out = self._branch(q, hca_k, hca_v, hca_mask, self.hca_sink.value, None)

    out = jnp.concatenate([csa_out, hca_out], axis=-1)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(
      b, s, self.n_heads * 2 * self.head_dim
    )
    return self.o_proj(out)


class DeepSeek4Block(nnx.Module):
  """Pre-norm transformer block: hybrid CSA+HCA attention + dense
  GeGLU FFN.
  """

  def __init__(  # noqa: PLR0913
    self,
    d_model,
    n_heads,
    n_kv_heads,
    head_dim,
    d_ff,
    *,
    csa_block_size,
    csa_top_k,
    hca_block_size,
    rngs,
  ):
    """Construct a DeepSeek-V4 transformer block.

    Args:
      d_model: model (residual stream) dimensionality.
      n_heads: number of query heads.
      n_kv_heads: number of key/value heads.
      head_dim: dimensionality of each attention head.
      d_ff: feed-forward hidden dimensionality.
      csa_block_size: CSA branch pooling block size.
      csa_top_k: number of CSA blocks attended to per query.
      hca_block_size: HCA branch pooling block size.
      rngs: random keys.
    """
    self.attn_norm = RMSNorm(d_model, rngs=rngs)
    self.attn = DeepSeek4Attention(
      d_model,
      n_heads,
      n_kv_heads,
      head_dim,
      csa_block_size=csa_block_size,
      csa_top_k=csa_top_k,
      hca_block_size=hca_block_size,
      rngs=rngs,
    )
    self.ffn_norm = RMSNorm(d_model, rngs=rngs)
    self.ffn = GeGLU(d_model, d_ff, rngs=rngs)

  def __call__(self, x: jax.Array, positions: jax.Array) -> jax.Array:
    """Apply the block.

    Args:
      x: input array, shape (batch, seq, d_model).
      positions: integer position ids, shape (batch, seq).

    Returns:
      jax.Array, same shape as x.
    """
    x = x + self.attn(self.attn_norm(x), positions)
    x = x + self.ffn(self.ffn_norm(x))
    return x


class DeepSeek4LLM(nnx.Module):
  """Decoder-only transformer in the DeepSeek-V4 architectural family:
  hybrid CSA+HCA attention with learnable attention sinks (see module
  docstring for the axes this reference implementation isolates versus
  real V4). Dense FFN only (no MoE -- that's MixtralSMoE's role in this
  suite; DeepSeekMLA covers the latent-KV-compression axis).
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
    *,
    csa_block_size=2,
    csa_top_k=4,
    hca_block_size=4,
    rngs,
  ):
    """Construct a DeepSeek4LLM.

    Args:
      vocab_size: token vocabulary size.
      d_model: model (residual stream) dimensionality.
      n_layers: number of transformer blocks.
      n_heads: number of query heads.
      n_kv_heads: number of key/value heads.
      head_dim: dimensionality of each attention head.
      d_ff: feed-forward hidden dimensionality.
      csa_block_size: CSA branch pooling block size.
      csa_top_k: number of CSA blocks attended to per query.
      hca_block_size: HCA branch pooling block size.
      rngs: random keys.
    """
    self.embed = nnx.Embed(
      vocab_size,
      d_model,
      embedding_init=nnx.with_partitioning(
        nnx.initializers.normal(), ("fsdp", None)
      ),
      rngs=rngs,
    )
    self.blocks = tuple(
      DeepSeek4Block(
        d_model,
        n_heads,
        n_kv_heads,
        head_dim,
        d_ff,
        csa_block_size=csa_block_size,
        csa_top_k=csa_top_k,
        hca_block_size=hca_block_size,
        rngs=rngs,
      )
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
      model, no MoE) -- kept for interface uniformity with MixtralSMoE
      so objective.py's causal_lm works unmodified across this suite.
    """
    hidden = self.embed(token_ids)
    for block in self.blocks:
      hidden = block(hidden, positions)
    hidden = self.final_norm(hidden)
    logits = self.lm_head(hidden)
    return logits, jnp.array(0.0)


def DeepSeek4(vocab_size, **kwargs):
  return DeepSeek4LLM(vocab_size, **kwargs)
