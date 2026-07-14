"""Built-in training hooks for use with train_fn's hooks= argument.

Every hook here matches train_fn's calling convention:
hook(step, *, model, optimizer, metrics) -> None, called every training
step. Hooks that don't need optimizer/metrics accept **kwargs to ignore
them.

Note: EMA state is not integrated with get_default_checkpointer -- saving
and restoring EMA state alongside model checkpoints is a separate,
larger change to checkpointer.py's save/restore item structure. Not
covered here.
"""

from collections.abc import Callable

import jax
from flax import nnx


def get_ema_hook(
  model: nnx.Module, decay: float = 0.999
) -> tuple[Callable, Callable]:
  """Construct an exponential-moving-average hook for weights.

  Args:
    model: the model from which the EMA state is initialized from
    decay: EMA decay rate

  Returns:
    a tuple (hook_fn, get_ema_model_fn):
      hook_fn(step, *, model, **kwargs) -> None: updates the weights
      get_ema_model_fn(model: nnx.Module) -> nnx.Module: returns a new,
        independent nnx.Module with the same structure as `model` but
        with the tracked EMA parameter values
  """
  _, ema_state = nnx.split(model)
  box = {"state": ema_state}

  def hook_fn(step, *, model, **kwargs):
    del step, kwargs
    _, state = nnx.split(model)
    box["state"] = jax.tree_util.tree_map(
      lambda ema, current: decay * ema + (1 - decay) * current,
      box["state"],
      state,
    )

  def get_ema_model_fn(model: nnx.Module) -> nnx.Module:
    graphdef, _ = nnx.split(model)
    return nnx.merge(graphdef, box["state"])

  return hook_fn, get_ema_model_fn
