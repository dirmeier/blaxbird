import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataloader
import jax
import optax
from absl import logging
from flax import nnx
from jax import random as jr
from jax.experimental import mesh_utils
from model import CNN, train_step, val_step

from blaxbird import get_default_checkpointer, train_fn


def get_optimizer(
  model, *, peak_lr=1e-4, n_steps, warmup_steps=1000, grad_clip_norm=1.0
):
  # warmup_cosine_decay_schedule requires decay_steps > warmup_steps (the
  # cosine phase runs over decay_steps - warmup_steps); clamp so a short
  # n_steps (e.g. a smoke run) doesn't raise ValueError from optax.
  warmup_steps = min(warmup_steps, n_steps // 2)
  schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=peak_lr,
    warmup_steps=warmup_steps,
    decay_steps=n_steps,
    end_value=peak_lr * 0.01,
  )
  tx = optax.chain(
    optax.clip_by_global_norm(grad_clip_norm), optax.adamw(schedule)
  )
  return nnx.Optimizer(model, tx=tx)


def get_sharding():
  num_devices = jax.local_device_count()
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((num_devices,)), ("data",)
  )
  model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
  data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
  return model_sharding, data_sharding


def metrics_hook(val_iter, hook_every_n_steps):
  def hook_fn(metrics, val_iter, hook_every_n_steps):
    def fn(step, *, model, **kwargs):
      if step % hook_every_n_steps != 0:
        return
      for _, batch in zip(range(5), val_iter):
        batch = next(iter(val_iter))
        logits = model(batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
          logits=logits, labels=batch["label"]
        ).mean()
        metrics.update(loss=loss, logits=logits, labels=batch["label"])
      if jax.process_index() == 0:
        curr_metrics = ", ".join(
          [f"{k}: {v}" for k, v in metrics.compute().items()]
        )
        logging.info(f"metrics at step {step}: {curr_metrics}")
        metrics.reset()

    return fn

  metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average("loss"),
  )
  return hook_fn(metrics, val_iter, hook_every_n_steps)


def get_hooks(val_itr, hook_every_n_steps):
  return [metrics_hook(val_itr, hook_every_n_steps)]


def get_train_and_val_itrs(rng_key, outfolder):
  return dataloader.data_loaders(
    rng_key,
    outfolder,
    split=["train[:90%]", "train[90%:]"],
    shuffle=[True, False],
  )


def run(n_steps, eval_every_n_steps, n_eval_batches):
  logging.set_verbosity(logging.INFO)
  outfolder = os.path.join(os.path.dirname(__file__), "workdir")

  train_itr, val_itr = get_train_and_val_itrs(
    jr.key(0), os.path.join(outfolder, "data")
  )

  model = CNN(rngs=nnx.rnglib.Rngs(jr.key(1)))
  optimizer = get_optimizer(model, n_steps=n_steps)

  save_fn, _, restore_last_fn = get_default_checkpointer(
    os.path.join(outfolder, "checkpoints"),
    save_every_n_steps=eval_every_n_steps,
  )
  hooks = get_hooks(val_itr, eval_every_n_steps) + [save_fn]

  model_sharding, data_sharding = get_sharding()
  optimizer = restore_last_fn(optimizer)

  train = train_fn(
    fns=(train_step, val_step),
    n_steps=n_steps,
    eval_every_n_steps=eval_every_n_steps,
    n_eval_batches=n_eval_batches,
    shardings=(model_sharding, data_sharding),
    hooks=hooks,
    log_to_wandb=False,
  )
  train(jr.key(2), optimizer, train_itr, val_itr)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-steps", type=int, default=1_000)
  parser.add_argument("--eval-every-n-steps", type=int, default=50)
  parser.add_argument("--n-eval-batches", type=int, default=10)
  args = parser.parse_args()
  run(args.n_steps, args.eval_every_n_steps, args.n_eval_batches)
