"""Smoke-test all three llm_reference model families on random data,
each under its own real simulated multi-device mesh.

No tokenizer, no real dataset, no actual training run beyond a handful
of steps on random tokens -- this proves each architecture, the shared
causal_lm objective, the shared generation loop, and (for the Mixtral
variant especially) real sharded dispatch/combine routing all compose
correctly under blaxbird.train_fn, not that any model learns anything.

Run with XLA_FLAGS="--xla_force_host_platform_device_count=8" to
exercise real multi-device sharding for all three (Gemma/DeepSeek use a
(2,4) fsdp+tp mesh built from all 8 simulated devices; Mixtral uses the
full (2,2,2) 3D mesh). Runs (unsharded, degenerate) on a single device
too, just without meaningfully exercising the sharding.

Note: train_fn logs train/val loss via absl.logging.info, which this
script does not raise verbosity for -- the printed generated-token-id
line per model family is the visible completion signal, not a loss line.
"""

import jax
import optax
from flax import nnx
from jax import numpy as jnp
from jax import random as jr
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P

from blaxbird import train_fn
from deepseek import DeepSeekMLA
from gemma import GemmaDense
from generate import generate
from mixtral import MixtralSMoE
from objective import causal_lm


def _random_batch_iter(vocab_size, batch, seq_len):
  key = jr.key(1)
  while True:
    key, batch_key = jr.split(key)
    token_ids = jr.randint(batch_key, (batch, seq_len + 1), 0, vocab_size)
    yield {"token_ids": token_ids[:, :-1], "target_ids": token_ids[:, 1:]}


def run_gemma(n_steps: int) -> None:
  n_devices = jax.local_device_count()
  fsdp = 2 if n_devices >= 4 else 1
  tp = n_devices // fsdp
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((fsdp, tp)), ("fsdp", "tp")
  )
  vocab_size = 100
  with mesh:
    model = GemmaDense(
      vocab_size,
      d_model=32,
      n_layers=4,
      n_heads=4,
      n_kv_heads=2,
      head_dim=8,
      d_ff=64,
      local_window=4,
      global_every=2,
      rngs=nnx.rnglib.Rngs(jr.key(0)),
    )
    optimizer = nnx.Optimizer(model, tx=optax.adamw(1e-3))
    fns = causal_lm()
    train = train_fn(
      fns=(fns.train_step, fns.val_step),
      n_steps=n_steps,
      eval_every_n_steps=max(1, n_steps // 2),
      n_eval_batches=1,
      mesh=mesh,
      data_partition_spec=P("fsdp"),
    )
    train(
      jr.key(2),
      optimizer,
      _random_batch_iter(vocab_size, 8, 16),
      _random_batch_iter(vocab_size, 8, 16),
    )
    prompt_ids = jnp.zeros((1, 4), dtype=jnp.int32)
    generated = generate(
      optimizer.model, jr.key(3), prompt_ids, max_new_tokens=8, max_seq_len=32
    )
    print(f"gemma: generated token ids {generated.tolist()}")


def run_deepseek(n_steps: int) -> None:
  n_devices = jax.local_device_count()
  fsdp = 2 if n_devices >= 4 else 1
  tp = n_devices // fsdp
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((fsdp, tp)), ("fsdp", "tp")
  )
  vocab_size = 100
  with mesh:
    model = DeepSeekMLA(
      vocab_size,
      d_model=32,
      n_layers=4,
      n_heads=4,
      d_latent=8,
      head_dim_nope=6,
      head_dim_rope=2,
      d_ff=64,
      rngs=nnx.rnglib.Rngs(jr.key(0)),
    )
    optimizer = nnx.Optimizer(model, tx=optax.adamw(1e-3))
    fns = causal_lm()
    train = train_fn(
      fns=(fns.train_step, fns.val_step),
      n_steps=n_steps,
      eval_every_n_steps=max(1, n_steps // 2),
      n_eval_batches=1,
      mesh=mesh,
      data_partition_spec=P("fsdp"),
    )
    train(
      jr.key(2),
      optimizer,
      _random_batch_iter(vocab_size, 8, 16),
      _random_batch_iter(vocab_size, 8, 16),
    )
    prompt_ids = jnp.zeros((1, 4), dtype=jnp.int32)
    generated = generate(
      optimizer.model, jr.key(3), prompt_ids, max_new_tokens=8, max_seq_len=32
    )
    print(f"deepseek: generated token ids {generated.tolist()}")


def run_mixtral(n_steps: int) -> None:
  n_devices = jax.local_device_count()
  if n_devices >= 8:
    fsdp, tp, expert = 2, 2, 2
  else:
    fsdp, tp, expert = 1, 1, n_devices
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((fsdp, tp, expert)), ("fsdp", "tp", "expert")
  )
  vocab_size = 100
  with mesh:
    model = MixtralSMoE(
      vocab_size,
      d_model=32,
      n_layers=4,
      n_heads=4,
      n_kv_heads=2,
      head_dim=8,
      d_ff=64,
      n_experts=4,
      n_active=2,
      rngs=nnx.rnglib.Rngs(jr.key(0)),
    )
    optimizer = nnx.Optimizer(model, tx=optax.adamw(1e-3))
    fns = causal_lm()
    train = train_fn(
      fns=(fns.train_step, fns.val_step),
      n_steps=n_steps,
      eval_every_n_steps=max(1, n_steps // 2),
      n_eval_batches=1,
      mesh=mesh,
      data_partition_spec=P("fsdp"),
    )
    train(
      jr.key(2),
      optimizer,
      _random_batch_iter(vocab_size, 8, 16),
      _random_batch_iter(vocab_size, 8, 16),
    )
    prompt_ids = jnp.zeros((1, 4), dtype=jnp.int32)
    generated = generate(
      optimizer.model, jr.key(3), prompt_ids, max_new_tokens=8, max_seq_len=32
    )
    print(f"mixtral: generated token ids {generated.tolist()}")


if __name__ == "__main__":
  run_gemma(n_steps=4)
  run_deepseek(n_steps=4)
  run_mixtral(n_steps=4)
