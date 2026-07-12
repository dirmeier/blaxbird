"""Sampled autoregressive generation, shared across every model family in
this suite. Recomputes the full prefix on every decoding step
rather than threading an explicit KV cache -- O(n^2) in sequence length,
the right tradeoff for reference/test code on tiny sequences, not a
production serving path (applies even to DeepSeekMLA, where a real cache
would normally be the point of the architecture -- see the design doc).
"""

import jax
import jax.numpy as jnp
from jax import random as jr


def generate(
  model,
  rng_key: jax.Array,
  prompt_ids: jax.Array,
  max_new_tokens: int,
  *,
  max_seq_len: int,
) -> jax.Array:
  """Autoregressively extend prompt_ids by max_new_tokens tokens.

  Args:
    model: any model in this suite (or anything with the same
      (token_ids, positions) -> (logits, aux_loss) signature).
    rng_key: a jax.random.key object.
    prompt_ids: integer token ids, shape (batch, prompt_len).
    max_new_tokens: number of tokens to generate.
    max_seq_len: total sequence length prompt_ids + generated tokens must
      not exceed.

  Returns:
    jax.Array, shape (batch, prompt_len + max_new_tokens).
  """
  batch, prompt_len = prompt_ids.shape
  total_len = prompt_len + max_new_tokens
  if total_len > max_seq_len:
    raise ValueError(
      f"prompt_len + max_new_tokens ({total_len}) exceeds "
      f"max_seq_len ({max_seq_len})"
    )

  tokens = prompt_ids
  for step in range(max_new_tokens):
    seq_len = tokens.shape[1]
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
    logits, _ = model(tokens, positions)
    next_logits = logits[:, -1, :]
    step_key = jr.fold_in(rng_key, step)
    next_token = jr.categorical(step_key, next_logits, axis=-1)
    tokens = jnp.concatenate([tokens, next_token[:, None]], axis=1)

  return tokens
