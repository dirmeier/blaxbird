from collections.abc import Callable, Iterable

import jax
from absl import logging
from flax import nnx
from jax import random as jr

try:
  import wandb
except ImportError:
  wandb = None


# ruff: noqa: ANN001, ANN202, ANN003
def _step_and_val_fns(fns):
  step_fn, eval_fn = fns

  def _train_step(model, rng_key, optimizer, metrics, batch, **kwargs):
    model.train()
    loss, grads = step_fn(model, rng_key, batch, **kwargs)
    optimizer.update(grads)
    metrics.update(loss=loss)
    return {"loss": loss}

  def _eval_step(model, rng_key, metrics, batch, **kwargs):
    model.eval()
    loss = eval_fn(model, rng_key, batch, **kwargs)
    metrics.update(loss=loss)
    return {"loss": loss}

  return _train_step, _eval_step


# ruff: noqa: PLW2901,PLR0913
def train_fn(
  *,
  fns: tuple[Callable, Callable],
  mesh: jax.sharding.Mesh | None = None,
  data_partition_spec: jax.sharding.PartitionSpec | None = None,
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
    mesh: a jax.sharding.Mesh to shard training over, or None to run
      unsharded on a single device. Per-parameter sharding is derived
      from each parameter's own nnx.with_partitioning annotation (see
      flax.nnx docs) via nnx.get_named_sharding -- parameters without
      such an annotation default to fully replicated, so passing a mesh
      with no annotated parameters gives plain data parallelism.
    data_partition_spec: how to shard each training/eval batch across
      `mesh`. Defaults to PartitionSpec() (fully replicated); pass e.g.
      PartitionSpec("data") to shard the batch dimension across a mesh
      axis named "data".
    n_steps: number of training/gradient steps
    eval_every_n_steps: specified how often to compute validation statistics.
    n_eval_batches: number of batches to use for validation
    log_to_wandb: whether to log results to wandb or not. Requires the
      optional `wandb` package to be installed.
    hooks: iterable of hooks

  Example:
    ```python
    import optax
    from flax import nnx
    from jax import random as jr

    model = CNN(rngs=nnx.rnglib.Rngs(jr.key(1)))
    optimizer = nnx.Optimizer(model, optax.adam(1e-4))

    train = train_fn(
      fns=(train_step, val_step),
      n_steps=100,
      eval_every_n_steps=10,
      n_eval_batches=10,
    )
    train(jr.key(2), optimizer, train_itr, val_itr)
    ```

  Raises:
    ImportError: if log_to_wandb is True but wandb is not installed.

  Returns:
    returns a callable for training
  """
  if log_to_wandb and wandb is None:
    raise ImportError(
      "log_to_wandb=True requires the 'wandb' package. Install it with "
      "`pip install wandb`."
    )
  _wandb = wandb
  _data_partition_spec = (
    data_partition_spec
    if data_partition_spec is not None
    else jax.sharding.PartitionSpec()
  )

  def train(
    rng_key: jax.Array,
    optimizer: nnx.Optimizer,
    train_itr: Iterable,
    val_itr: Iterable,
  ) -> None:
    """Train a NNX model.

    Args:
      rng_key: a jax.random.key object
      optimizer: a nnx.Optimizer object. The wrapped model (optimizer.model)
        is trained in place -- there is no separate model argument, since
        nnx.Optimizer already owns the model it wraps.
      train_itr: an infinite data loader, i.e., an iteratlor that keeps running.
        You can, for instance, construct this as a tfds.NumpyIterator or a
        grain.DataLoader.
      val_itr: an infinite data loader, i.e., an iteratlor that keeps running.
        You can, for instance, construct this as a tfds.NumpyIterator or a
        grain.DataLoader.
    """
    model = optimizer.model
    # get train and val fns
    step_fn, eval_fn = _step_and_val_fns(fns)
    # get model and shard
    if mesh is not None:
      state = nnx.state((model, optimizer))
      sharding = nnx.get_named_sharding(state, mesh)
      state = jax.device_put(state, sharding)
      nnx.update((model, optimizer), state)
    # metrics
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    metrics_history = {}
    # run training
    step_key, rng_key = jr.split(rng_key)
    for step, batch in zip(range(1, n_steps + 1), train_itr, strict=False):
      train_key, val_key = jr.split(jr.fold_in(step_key, step))
      if mesh is not None:
        batch = jax.device_put(
          batch, jax.NamedSharding(mesh, _data_partition_spec)
        )
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
          # nnx.MultiMetric.compute()'s Metric return type is a stub
          # imprecision -- the runtime value is a scalar array.
          metrics_history[f"train/{metric}"] = float(
            value  # type: ignore[arg-type]
          )
        # do evaluation loop
        for val_idx, batch in zip(range(n_eval_batches), val_itr, strict=False):
          if mesh is not None:
            batch = jax.device_put(
              batch, jax.NamedSharding(mesh, _data_partition_spec)
            )
          eval_fn(
            model=model,
            rng_key=jr.fold_in(val_key, val_idx),
            metrics=metrics,
            batch=batch,
          )
        # store val losses
        for metric, value in metrics.compute().items():
          # nnx.MultiMetric.compute()'s Metric return type is a stub
          # imprecision -- the runtime value is a scalar array.
          metrics_history[f"val/{metric}"] = float(
            value  # type: ignore[arg-type]
          )
        metrics.reset()
        # log losses after each val round
        if jax.process_index() == 0:
          logging.info(
            f"loss at step {step}: "
            f"{metrics_history['train/loss']}/"
            f"{metrics_history['val/loss']}"
          )
        if log_to_wandb and jax.process_index() == 0:
          assert _wandb is not None  # validated in train_fn above
          _wandb.log(metrics_history, step=step)
      for h in hooks:
        h(step, model=model, optimizer=optimizer, metrics=metrics_history)

  return train
