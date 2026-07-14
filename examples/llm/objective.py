"""LM training objectives.

Contains train_step, eval_step and generate functions.
"""

import optax
from flax import nnx
import jax
from jax import numpy as jnp

import jax.numpy as jnp
from jax import random as jr


def get_training_and_eval_fns(aux_loss_coef: float = 0.01):
  """Construct causal LM train/val step functions.

  Args:
    aux_loss_coef: weight applied to the model's own aux_loss

  Returns:
    train_step and eval_step functions
  """

  def _loss_fn(model, rng_key, batch):
    del rng_key
    seq_len = batch["token_ids"].shape[1]
    positions = jnp.broadcast_to(jnp.arange(seq_len), batch["token_ids"].shape)
    logits, aux_loss = model(batch["token_ids"], positions)
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch["target_ids"]
    ).mean()
    return ce_loss + aux_loss_coef * aux_loss

  def train_step(model, rng_key, batch, **kwargs):
    del kwargs
    return nnx.value_and_grad(_loss_fn)(model, rng_key, batch)

  def val_step(model, rng_key, batch, **kwargs):
    del kwargs
    return _loss_fn(model, rng_key, batch)

  return train_step, val_step

def sample(
  model,
  rng_key: jax.Array,
  prompt: jax.Array,
  max_new_tokens: int,
  *,
  max_seq_len: int,
) -> jax.Array:
  """Generate new tokens.

  Args:
    model: a LM
    rng_key: a jax.random.key object
    prompt: integer token ids, shape (batch, prompt_len)
    max_new_tokens: number of tokens to generate
    max_seq_len: total sequence length prompt + generated tokens must
      not exceed

  Returns:
    jax.Array, shape (batch, prompt_len + max_new_tokens)
  """
  batch, prompt_len = prompt.shape
  total_len = prompt_len + max_new_tokens
  if total_len > max_seq_len:
    raise ValueError(
      f"prompt_len + max_new_tokens ({total_len}) exceeds "
      f"max_seq_len ({max_seq_len})"
    )

  tokens = prompt
  for step in range(max_new_tokens):
    seq_len = tokens.shape[1]
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))
    logits, _ = model(tokens, positions)
    next_logits = logits[:, -1, :]
    next_token = jr.categorical(jr.fold_in(rng_key, step), next_logits, axis=-1)
    tokens = jnp.concatenate([tokens, next_token[:, None]], axis=1)

  return tokens
