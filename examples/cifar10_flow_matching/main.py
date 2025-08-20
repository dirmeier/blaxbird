import os

import dataloader
import jax
import optax
from absl import logging
from flax import nnx
from jax import random as jr
from jax.experimental import mesh_utils

from blaxbird import get_default_checkpointer, train_fn


def get_optimizer(model, lr=1e-4):
  tx = optax.adamw(lr)
  tx = nnx.Optimizer(model, tx=tx)
  return tx


def get_sharding():
  num_devices = jax.local_device_count()
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((num_devices,)), ("data",)
  )
  model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
  data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
  return model_sharding, data_sharding


def visualize_hook(val_iter, eval_every_n_steps):
  def hook_fn(metrics, val_iter, eval_every_n_steps):
    def fn(step, *, model, **kwargs):
      if step % eval_every_n_steps != 0:
        return
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
  return hook_fn(metrics, val_iter, eval_every_n_steps)


def get_hooks(val_itr, eval_every_n_steps):
  return [visualize_hook(val_itr, eval_every_n_steps)]


def get_train_and_val_itrs(rng_key, outfolder):
  return dataloader.data_loaders(
    rng_key, outfolder, split=["train[:90%]", "train[90%:]"]
  )


def run():
  logging.set_verbosity(logging.INFO)

  outfolder = os.path.dirname(__file__)
  n_steps, eval_every_n_steps, n_eval_batches = 10_000, 100, 20
  train_itr, val_itr = get_train_and_val_itrs(
    jr.key(0), os.path.join(outfolder, "data")
  )

  model = CNN(rngs=nnx.rnglib.Rngs(jr.key(1)))
  optimizer = get_optimizer(model)
  model_sharding, data_sharding = get_sharding()
  hooks = get_hooks(val_itr, eval_every_n_steps)

  save_fn, _, restore_last_fn = get_default_checkpointer(
    os.path.join(outfolder, "checkpoints"), save_every_n_steps=100
  )
  model, optimizer = restore_last_fn(model, optimizer)
  hooks.append(save_fn)

  train = train_fn(
    fns=(train_step, val_step),
    n_steps=n_steps,
    eval_every_n_steps=eval_every_n_steps,
    n_eval_batches=n_eval_batches,
    shardings=(model_sharding, data_sharding),
    hooks=hooks,
    log_to_wandb=False,
  )
  train(jr.key(2), model, optimizer, train_itr, val_itr)


if __name__ == "__main__":
  run()
