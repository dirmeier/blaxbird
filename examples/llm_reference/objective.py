"""Causal-language-model training objective, shared across every model
family in this suite (GemmaDense, DeepSeekMLA, MixtralSMoE) -- works
uniformly since every model's __call__ returns (logits, aux_loss), with
dense models always returning aux_loss=0.0.
"""

import optax
from flax import nnx
from jax import numpy as jnp

from blaxbird._src._types import ObjectiveFns


def causal_lm(aux_loss_coef: float = 0.01) -> ObjectiveFns:
  """Construct next-token-prediction train/val step functions.

  Args:
    aux_loss_coef: weight applied to the model's own aux_loss (already
      pre-weighted for MixtralSMoE, always 0.0 for the dense models)
      before adding it to the cross-entropy loss. For MixtralSMoE this
      compounds with MixtralLLM's own aux_loss_coef constructor argument
      (default 0.01 there too) -- the net effective weight on the raw
      load-balancing loss is this value times that one (0.01 * 0.01 =
      1e-4 with both defaults), not this value alone. Intentionally not
      un-compounded here: doing so would require this function to
      special-case MixtralSMoE, breaking the point of this objective
      being agnostic to which model family it's given.

  Returns:
    an ObjectiveFns with sample_fn=None -- generation is a separate loop
    (see generate.py), not a fit for the single-shot sample_fn contract
    other blaxbird objectives use.
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

  return ObjectiveFns(train_step=train_step, val_step=val_step, sample_fn=None)
