import argparse
import os

import dataloader
import jax
import optax
from examples.llm.nn.deepseek import DeepSeek4
from flax import nnx
from examples.llm.nn.gemma import GemmaDense
from jax import numpy as jnp
from jax import random as jr
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
from examples.llm.nn.qwen import Qwen3NextHybrid
from objective import get_training_and_eval_fns, sample

from blaxbird import train_fn

_DATA_DIR = os.path.join(os.path.dirname(__file__), "workdir", "data")


def _get_train_and_val_itrs(rng_key, *, seq_len, batch_size):
  return dataloader.data_loaders(
    rng_key, _DATA_DIR, seq_len=seq_len, batch_size=batch_size
  )


def run_gemma(n_steps: int) -> None:
  n_devices = jax.local_device_count()
  fsdp = 2 if n_devices >= 4 else 1
  tp = n_devices // fsdp
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((fsdp, tp)), ("fsdp", "tp")
  )
  with mesh:
    model = GemmaDense(
      dataloader.VOCAB_SIZE,
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
    fns = get_training_and_eval_fns()
    train = train_fn(
      fns=fns,
      n_steps=n_steps,
      eval_every_n_steps=max(1, n_steps // 2),
      n_eval_batches=1,
      mesh=mesh,
      data_partition_spec=P("fsdp"),
    )
    train_itr, val_itr = _get_train_and_val_itrs(
      jr.key(1), seq_len=32, batch_size=8
    )
    train(jr.key(2), optimizer, train_itr, val_itr)
    prompt_ids = jnp.zeros((1, 4), dtype=jnp.int32)
    generated = sample(
      optimizer.model, jr.key(3), prompt_ids, max_new_tokens=8, max_seq_len=32
    )
    text = bytes(int(b) for b in generated[0]).decode("utf-8", errors="replace")
    print(f"gemma: generated text {text!r}")


def run_deepseek4(n_steps: int) -> None:
  n_devices = jax.local_device_count()
  fsdp = 2 if n_devices >= 4 else 1
  tp = n_devices // fsdp
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((fsdp, tp)), ("fsdp", "tp")
  )
  with mesh:
    model = DeepSeek4(
      dataloader.VOCAB_SIZE,
      d_model=32,
      n_layers=4,
      n_heads=4,
      n_kv_heads=2,
      head_dim=8,
      d_ff=64,
      csa_block_size=2,
      csa_top_k=4,
      hca_block_size=4,
      rngs=nnx.rnglib.Rngs(jr.key(0)),
    )
    optimizer = nnx.Optimizer(model, tx=optax.adamw(1e-3))
    fns = get_training_and_eval_fns()
    train = train_fn(
      fns=fns,
      n_steps=n_steps,
      eval_every_n_steps=max(1, n_steps // 2),
      n_eval_batches=1,
      mesh=mesh,
      data_partition_spec=P("fsdp"),
    )
    train_itr, val_itr = _get_train_and_val_itrs(
      jr.key(1), seq_len=32, batch_size=8
    )
    train(jr.key(2), optimizer, train_itr, val_itr)
    prompt_ids = jnp.zeros((1, 4), dtype=jnp.int32)
    generated = sample(
      optimizer.model, jr.key(3), prompt_ids, max_new_tokens=8, max_seq_len=32
    )
    text = bytes(int(b) for b in generated[0]).decode("utf-8", errors="replace")
    print(f"deepseek4: generated text {text!r}")


def run_qwen3_next(n_steps: int) -> None:
  n_devices = jax.local_device_count()
  if n_devices >= 8:
    fsdp, tp, expert = 2, 2, 2
  else:
    fsdp, tp, expert = 1, 1, n_devices
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((fsdp, tp, expert)), ("fsdp", "tp", "expert")
  )
  with mesh:
    model = Qwen3NextHybrid(
      dataloader.VOCAB_SIZE,
      d_model=32,
      n_layers=4,
      n_heads=4,
      n_kv_heads=2,
      head_dim=8,
      d_ff=64,
      n_experts=8,
      n_active=2,
      rngs=nnx.rnglib.Rngs(jr.key(0)),
    )
    optimizer = nnx.Optimizer(model, tx=optax.adamw(1e-3))
    fns = get_training_and_eval_fns()
    train = train_fn(
      fns=fns,
      n_steps=n_steps,
      eval_every_n_steps=max(1, n_steps // 2),
      n_eval_batches=1,
      mesh=mesh,
      data_partition_spec=P("fsdp"),
    )
    train_itr, val_itr = _get_train_and_val_itrs(
      jr.key(1), seq_len=32, batch_size=8
    )
    train(jr.key(2), optimizer, train_itr, val_itr)
    prompt_ids = jnp.zeros((1, 4), dtype=jnp.int32)
    generated = sample(
      optimizer.model, jr.key(3), prompt_ids, max_new_tokens=8, max_seq_len=32
    )
    text = bytes(int(b) for b in generated[0]).decode("utf-8", errors="replace")
    print(f"qwen3_next: generated text {text!r}")


_RUNS = {
  "gemma4": run_gemma,
  "deepseek4": run_deepseek4,
  "qwen3next": run_qwen3_next,
}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", choices=sorted(_RUNS), default=None)
  parser.add_argument("--n-steps", type=int, default=100)
  args = parser.parse_args()

  runs = [_RUNS[args.model]] if args.model else list(_RUNS.values())
  for run in runs:
    run(n_steps=args.n_steps)


if __name__ == "__main__":
  main()
