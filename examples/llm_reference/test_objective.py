import jax.numpy as jnp
from flax import nnx
from jax import random as jr

from deepseek import DeepSeekMLA
from gemma import GemmaDense
from mixtral import MixtralSMoE
from objective import causal_lm


def _batch():
  return {
    "token_ids": jnp.zeros((2, 6), dtype=jnp.int32),
    "target_ids": jnp.ones((2, 6), dtype=jnp.int32),
  }


def test_causal_lm_train_step_runs_for_all_three_model_families():
  fns = causal_lm()
  models = [
    GemmaDense(
      vocab_size=50,
      d_model=32,
      n_layers=2,
      n_heads=4,
      n_kv_heads=2,
      head_dim=8,
      d_ff=64,
      local_window=4,
      global_every=2,
      rngs=nnx.rnglib.Rngs(jr.key(0)),
    ),
    DeepSeekMLA(
      vocab_size=50,
      d_model=32,
      n_layers=2,
      n_heads=4,
      d_latent=8,
      head_dim_nope=6,
      head_dim_rope=2,
      d_ff=64,
      rngs=nnx.rnglib.Rngs(jr.key(0)),
    ),
    MixtralSMoE(
      vocab_size=50,
      d_model=32,
      n_layers=2,
      n_heads=4,
      n_kv_heads=2,
      head_dim=8,
      d_ff=64,
      n_experts=4,
      n_active=2,
      rngs=nnx.rnglib.Rngs(jr.key(0)),
    ),
  ]
  batch = _batch()
  for model in models:
    loss, grads = fns.train_step(model, jr.key(1), batch)
    assert loss.shape == ()
  assert fns.sample_fn is None


def test_causal_lm_val_step_runs():
  fns = causal_lm()
  model = GemmaDense(
    vocab_size=50,
    d_model=32,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    head_dim=8,
    d_ff=64,
    local_window=4,
    global_every=2,
    rngs=nnx.rnglib.Rngs(jr.key(0)),
  )
  loss = fns.val_step(model, jr.key(1), _batch())
  assert loss.shape == ()
