import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataloader
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from absl import logging
from flax import nnx
from jax import random as jr
from jax.experimental import mesh_utils

from blaxbird import get_default_checkpointer, train_fn
from _common import rfm
from _common.nn import dit  # for getattr(dit, dit_type, ...) below


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


def get_mesh():
  num_devices = jax.local_device_count()
  return jax.sharding.Mesh(
    mesh_utils.create_device_mesh((num_devices,)), ("data",)
  )


def visualize_hook(sample_fn, val_iter, hook_every_n_steps, log_to_wandb):
  n_row, n_col, img_size = 12, 32, (32, 32, 3)

  def convert_batch_to_image_grid(image_batch):
    reshaped = (
      image_batch.reshape(n_row, n_col, *img_size)
      .transpose([0, 2, 1, 3, 4])
      .reshape(n_row * img_size[0], n_col * img_size[1], img_size[2])
    )
    return (reshaped + 1.0) / 2.0

  def plot(images):
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(
      images,
      interpolation="nearest",
      cmap="gray",
    )
    plt.axis("off")
    plt.tight_layout()
    return fig

  def fn(step, *, model, **kwargs):
    if step % hook_every_n_steps != 0:
      return
    all_samples = []
    for i, batch in enumerate(val_iter):
      samples = sample_fn(
        model,
        jr.fold_in(jr.key(step), i),
        sample_shape=batch["inputs"].shape,
        context=batch["context"],
      )
      all_samples.append(samples)
      if len(all_samples) * all_samples[0].shape[0] >= n_row * n_col:
        break
    all_samples = np.concatenate(all_samples, axis=0)[: (n_row * n_col)]
    all_samples = convert_batch_to_image_grid(all_samples)
    fig = plot(all_samples)
    if jax.process_index() == 0 and log_to_wandb:
      wandb.log({"images": wandb.Image(fig)}, step=step)

  return fn


def get_hooks(sample_fn, val_itr, hook_every_n_steps, log_to_wandb):
  return [visualize_hook(sample_fn, val_itr, hook_every_n_steps, log_to_wandb)]


def get_train_and_val_itrs(rng_key, outfolder):
  return dataloader.data_loaders(
    rng_key,
    outfolder,
    split=["train[:90%]", "train[90%:]"],
    shuffle=[True, False],
  )


def run(n_steps, eval_every_n_steps, n_eval_batches, dit_type, log_to_wandb):
  logging.set_verbosity(logging.INFO)
  outfolder = os.path.join(os.path.dirname(__file__), "workdir")

  train_itr, val_itr = get_train_and_val_itrs(
    jr.key(0), os.path.join(outfolder, "data")
  )

  model = getattr(dit, dit_type)(
    image_size=(32, 32, 3), n_classes=10, rngs=nnx.rnglib.Rngs(jr.key(1))
  )
  objective = rfm()
  optimizer = get_optimizer(model, n_steps=n_steps)

  save_fn, _, restore_last_fn = get_default_checkpointer(
    os.path.join(outfolder, "checkpoints"),
    save_every_n_steps=eval_every_n_steps,
  )
  hooks = get_hooks(
    objective.sample_fn, val_itr, eval_every_n_steps, log_to_wandb
  ) + [save_fn]

  mesh = get_mesh()
  optimizer = restore_last_fn(optimizer)

  train = train_fn(
    fns=(objective.train_step, objective.val_step),
    n_steps=n_steps,
    eval_every_n_steps=eval_every_n_steps,
    n_eval_batches=n_eval_batches,
    mesh=mesh,
    data_partition_spec=jax.sharding.PartitionSpec("data"),
    hooks=hooks,
    log_to_wandb=False,
  )
  train(jr.key(2), optimizer, train_itr, val_itr)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-steps", type=int, default=1_000)
  parser.add_argument("--eval-every-n-steps", type=int, default=50)
  parser.add_argument("--n-eval-batches", type=int, default=10)
  parser.add_argument(
    "--dit", type=str, choices=["SmallDiT", "BaseDiT"], default="SmallDiT"
  )
  parser.add_argument("--log-to-wandb", action="store_true")
  args = parser.parse_args()
  run(
    args.n_steps,
    args.eval_every_n_steps,
    args.n_eval_batches,
    args.dit,
    args.log_to_wandb,
  )
