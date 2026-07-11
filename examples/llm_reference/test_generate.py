import jax.numpy as jnp
from flax import nnx
from jax import random as jr

from gemma import GemmaDense
from generate import generate


def _model():
  return GemmaDense(
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


def test_generate_extends_prompt_by_max_new_tokens():
  model = _model()
  prompt_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
  out = generate(model, jr.key(1), prompt_ids, max_new_tokens=4, max_seq_len=16)
  assert out.shape == (1, 3 + 4)
  assert jnp.array_equal(out[:, :3], prompt_ids)


def test_generate_is_deterministic_given_same_key():
  model = _model()
  prompt_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
  out1 = generate(
    model, jr.key(1), prompt_ids, max_new_tokens=4, max_seq_len=16
  )
  out2 = generate(
    model, jr.key(1), prompt_ids, max_new_tokens=4, max_seq_len=16
  )
  assert jnp.array_equal(out1, out2)
