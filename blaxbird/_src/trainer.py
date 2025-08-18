from collections.abc import Callable, Iterable

import grain.python as grain
import jax
import wandb
from absl import logging
from flax import nnx
from jax import random as jr


# ruff: noqa: ANN001, ANN202, ANN003
def _step_and_val_fns(fns):
  step, eval = fns

  def _train_step(model, rng_key, optimizer, metrics, batch, **kwargs):
    model.train()
    loss, grads = step(model, rng_key, batch, **kwargs)
    optimizer.update(grads)
    metrics.update(loss=loss)
    return {"loss": loss}

  def _eval_step(model, rng_key, metrics, batch, **kwargs):
    model.eval()
    loss = eval(model, rng_key, batch, **kwargs)
    metrics.update(loss=loss)
    return {"loss": loss}

  return _train_step, _eval_step


# ruff: noqa: PLW2901,PLR0913
def train_fn(
  *,
  fns: tuple[Callable, Callable],
  shardings: tuple[jax.NamedSharding, jax.NamedSharding],
  n_steps: int,
  eval_every_n_steps: int,
  n_eval_batches: int,
  log_to_wandb: bool = False,
  hooks: Iterable[Callable] = (),
) -> Callable:
  """Construct a function to train NNX models.

  Args:
    fns: a tuple of two callables. The first one is used as a step function
      , i.e., function to do gradient steps. The second one is used as an
      validation function.
    shardings: a tuple of shardings, the first one for the model, the second
      one for the data.
    n_steps: number of training/gradient steps
    eval_every_n_steps: specified how often to compute validation statistics.
    n_eval_batches: number of batches to use for validation
    log_to_wandb: whether to log results to wandb or not
    hooks: iterable of hooks

  Returns:
    returns a callable for training
  """

  def train(
    rng_key: jax.Array,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_itr: grain.DataLoader,
    val_itr: grain.DataLoader,
  ) -> None:
    """Train a NNX model.

    Args:
      rng_key: a jax.random.key object
      model: a NNX model
      optimizer: a nnx.Optimizer object
      train_itr: a data laoder
      val_itr: a data laoder
    """
    # get train and val fns
    step_fn, eval_fn = _step_and_val_fns(fns)
    # get model and replicate
    state = nnx.state((model, optimizer))
    state = jax.device_put(state, shardings[0])
    nnx.update((model, optimizer), state)
    # metrics
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    metrics_history = {}
    # run training
    step_key, rng_key = jr.split(rng_key)
    for step, batch in zip(range(1, n_steps + 1), train_itr):
      train_key, val_key = jr.split(jr.fold_in(step_key, step))
      batch = jax.device_put(batch, shardings[1])
      # do a gradient step
      step_fn(
        model=model,
        rng_key=train_key,
        optimizer=optimizer,
        metrics=metrics,
        batch=batch,
      )
      is_first_step = step == 1
      is_last_step = step == n_steps
      is_first_or_last_step = is_first_step or is_last_step
      if step % eval_every_n_steps == 0 or is_first_or_last_step:
        # store training losses
        for metric, value in metrics.compute().items():
          metrics_history[f"train/{metric}"] = float(value)
        # do evaluation loop
        for val_idx, batch in zip(range(n_eval_batches), val_itr):
          batch = jax.device_put(batch, shardings[1])
          eval_fn(
            model=model,
            rng_key=jr.fold_in(val_key, val_idx),
            metrics=metrics,
            batch=batch,
          )
        # store val losses
        for metric, value in metrics.compute().items():
          metrics_history[f"val/{metric}"] = float(value)
        metrics.reset()
        # log losses after each val round
        if jax.process_index() == 0:
          logging.info(
            f"loss at step {step}: "
            f"{metrics_history['train/loss']}/"
            f"{metrics_history['val/loss']}"
          )
        if log_to_wandb and jax.process_index() == 0:
          wandb.log(metrics_history, step=step)
      for h in hooks:
        h(step, model=model, optimizer=optimizer, metrics=metrics_history)

  return train
