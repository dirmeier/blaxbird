import os
from collections.abc import Callable

import orbax.checkpoint as ocp
from absl import logging
from flax import nnx


def get_default_checkpointer(
  outfolder: str,
  *,
  save_every_n_steps: int,
  max_to_keep: int = 5,
) -> tuple[Callable, Callable, Callable]:
  """Construct functions for checkpointing functionality.

  Args:
    outfolder: a path specifying where checkpoints are stored
    save_every_n_steps: how often to store checkpoints
    max_to_keep: number of checkpoints to store before they get deleted

  Returns:
    returns function to saev and restore checkpoints
  """
  checkpointer = ocp.PyTreeCheckpointer()
  options = ocp.CheckpointManagerOptions(
    max_to_keep=max_to_keep,
    create=True,
    best_mode="min",
    best_fn=lambda x: x["val/loss"],
  )
  checkpoint_manager = ocp.CheckpointManager(
    os.path.join(outfolder, "best"),
    options=options,
    item_names=("state", "opt_state"),
  )

  def save_fn(
    step: int, *, model: nnx.Module, optimizer: nnx.Optimizer, metrics: dict
  ) -> None:
    """Save a checkpoint.

    Args:
       step: number of training/gradient step
       model: a nnx.Model object
       optimizer: a nnx.Optimizer object
       metrics: a dictionary of metrics. One of which must be the criterion
        specified above.
    """
    if step % save_every_n_steps != 0:
      return

    _, state = nnx.split(model)
    _, opt_state = nnx.split(optimizer.opt_state)
    try:
      logging.info("saving best checkpoint")
      checkpoint_manager.save(
        step=step,
        args=ocp.args.Composite(
          state=ocp.args.StandardSave(state),
          opt_state=ocp.args.StandardSave(opt_state),
        ),
        metrics=metrics,
        force=True,
      )
      checkpoint_manager.wait_until_finished()
    except Exception as e:
      logging.error(f"could not save better checkpoint because of: {e}")
      logging.error("resuming nonetheless")
    try:
      logging.info("saving last checkpoint")
      checkpointer.save(
        os.path.join(outfolder, "last"), (state, opt_state), force=True
      )
    except Exception as e:
      logging.error(f"could not last checkpoint because of: {e}")
      logging.error("resuming nonetheless")

  def restore_best_fn(optimizer: nnx.Optimizer) -> nnx.Optimizer:
    """Restore the best checkpoint.

    Args:
     optimizer: a nnx.Optimizer object. Its wrapped model (optimizer.model)
      and opt_state are both updated in place and returned on the same
      optimizer instance.
    """
    graph_def, state = nnx.split(optimizer.model)
    opt_def, opt_state = nnx.split(optimizer.opt_state)
    try:
      logging.info("trying to restore best checkpoint")
      restored = checkpoint_manager.restore(
        checkpoint_manager.best_step(),
        args=ocp.args.Composite(
          state=ocp.args.StandardRestore(nnx.eval_shape(lambda: state)),
          opt_state=ocp.args.StandardRestore(nnx.eval_shape(lambda: opt_state)),
        ),
      )
      optimizer.model = nnx.merge(graph_def, restored["state"])
      optimizer.opt_state = nnx.merge(opt_def, restored["opt_state"])
    except FileNotFoundError:
      logging.warning("could not find checkpoint. resuming with blank state")
    return optimizer

  def restore_last_fn(optimizer: nnx.Optimizer) -> nnx.Optimizer:
    """Restore the latest training checkpoint.

    Args:
     optimizer: a nnx.Optimizer object. Its wrapped model (optimizer.model)
      and opt_state are both updated in place and returned on the same
      optimizer instance.
    """
    graphdef, state = nnx.split(optimizer.model)
    optdef, opt_state = nnx.split(optimizer.opt_state)
    state = nnx.eval_shape(lambda: state)
    opt_state = nnx.eval_shape(lambda: opt_state)
    try:
      logging.info("trying to restore last checkpoint")
      restored = checkpointer.restore(
        os.path.join(outfolder, "last"), (state, opt_state)
      )
      optimizer.model = nnx.merge(graphdef, restored[0])
      optimizer.opt_state = nnx.merge(optdef, restored[1])
    except FileNotFoundError:
      logging.warning("could not find checkpoint. resuming with blank state")
    return optimizer

  return save_fn, restore_best_fn, restore_last_fn
