# blaxbird [blækbɜːd]

[![ci](https://github.com/dirmeier/blaxbird/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/blaxbird/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/blaxbird.svg?colorB=black&style=flat)](https://pypi.org/project/blaxbird/)

> A high-level API to build and train NNX models

`Blaxbird` [blækbɜːd] is a high-level API to easily build NNX models and train them on CPU or GPU.

Using `blaxbird` one can
- concisely define models and loss functions without the usual JAX/Flax verbosity,
- easily define checkpointers that save the best and most current network weights,
- distribute data and model weights over multiple processes or GPUs,
- define hooks that are periodically called during training.

## Quickstart

To use `blaxbird`, one only needs to define a model, a loss function, and train and validation step functions:
```python
import optax
from flax import nnx

class CNN(nnx.Module):
  ...

def loss_fn(model, images, labels):
  logits = model(images)
  return optax.losses.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=labels
  ).mean()

def train_step(model, rng_key, batch):
    return nnx.value_and_grad(loss_fn)(model, batch["image"], batch["label"])

def val_step(model, rng_key, batch):
    return loss_fn(model, batch["image"], batch["label"])
```

You can then define construct (and use) a training function like this:

```python
import optax
from flax import nnx
from jax import random as jr

from blaxbird import train_fn

model = CNN(rngs=nnx.rnglib.Rngs(jr.key(1)))
optimizer = nnx.Optimizer(model, optax.adam(1e-4))

train = train_fn(
  fns=(train_step, val_step),
  n_steps=100,
  eval_every_n_steps=10,
  n_eval_batches=10
)
train(jr.key(2), optimizer, train_itr, val_itr)
```

## Examples

Full self-contained examples can be found in [examples](examples/).

## Installation

To install the package from PyPI, call:

```bash
pip install blaxbird
```

To install the latest GitHub <RELEASE>, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/blaxbird@<RELEASE>
```

## API

`train_fn` is a higher order function with the following signature:

```python
def train_fn(
  *,
  fns: tuple[Callable, Callable],
  mesh: jax.sharding.Mesh | None = None,
  data_partition_spec: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(),
  n_steps: int,
  eval_every_n_steps: int,
  n_eval_batches: int,
  log_to_wandb: bool = False,
  hooks: Iterable[Callable] = (),
) -> Callable:
  ...
```

The returned `train` callable has signature `train(rng_key, optimizer, train_itr, val_itr) -> None` -- it derives the model from `optimizer.model`, so there is no separate `model` argument.

We briefly explain the more ambiguous argument types below.

### `fns`

`fns` is a required argument consistenf of tuple of two functions, a step function and a validation function.
In the simplest case they look like this:

```python
def train_step(model, rng_key, batch):
    return nnx.value_and_grad(loss_fn)(model, batch["image"], batch["label"])

def val_step(model, rng_key, batch):
    return loss_fn(model, batch["image"], batch["label"])
```

Both `train_step` and `val_step` have the same arguments and argument types:
- `model` specifies a `nnx.Module`, i.e., a neural network like the CNN shown above.
- `rng_key` is a `jax.random.key` in case you need to generate random numbers.
- `batch` is a sample from a data loader (to be specified later).

The loss function that is called by both computes a *scalar* loss value. B
While `train_step` returns has to return the loss and gradients, `val_step` only needs
to return the loss.

### `mesh` and `data_partition_spec`

To specify how data and model weights are distributed over devices and processes,
`blaxbird` uses JAX' [sharding](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) functionality.

`mesh` is a `jax.sharding.Mesh` describing your device topology. Per-parameter sharding is derived from each parameter's own `nnx.with_partitioning`
annotation (see the [flax.nnx docs](https://flax.readthedocs.io/en/latest/)) via `nnx.get_named_sharding` -- parameters without such an annotation default to fully replicated, so a mesh with no annotated parameters gives plain data parallelism. `data_partition_spec` controls how each training/eval batch is sharded across `mesh` (defaults to `PartitionSpec()`, fully replicated). You can, if you don't want to distribute anything, just leave `mesh` as `None` or not specify it.

An example is shown below, sharding only the data over `num_devices` devices
(the model has no `with_partitioning` annotations, so it stays fully
replicated):

```python
def get_mesh():
  num_devices = jax.local_device_count()
  return jax.sharding.Mesh(
    mesh_utils.create_device_mesh((num_devices,)), ("data",)
  )

mesh = get_mesh()
```

Pass `mesh=mesh, data_partition_spec=jax.sharding.PartitionSpec("data")` to
`train_fn`. For real FSDP/tensor-parallel sharding, annotate your model's
layers with `nnx.with_partitioning` -- see
[examples/fsdp_tp_demo](examples/fsdp_tp_demo) for a worked 2D-mesh example.

### `hooks`

`hooks` is a list of callables which are periodically called during training.
Each hook has to have the following signature:

```python
def hook_fn(step, *, model, **kwargs) -> None:
  ...
```

It takes an integer `step` specifying the current training iteration and the model itself.
For instance, if you want to track custom metrics during validation, you could create a hook like this:

```python
def hook_fn(metrics, val_iter, hook_every_n_steps):
  def fn(step, *, model, **kwargs):
    if step % hook_every_n_steps != 0:
      return
    for batch in val_iter:
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
hook = hook_fn(metrics, val_iter, hook_every_n_steps)
```

This creates a hook function `hook` that after `eval_every_n_steps` steps iterates over the validation set
computes accuracy and loss, and then logs everything.

To provide multiple hooks to the train function, just concatenate them in a list.

#### A checkpointing `hook`

We provide a convenient hook for checkpointing which can be constructed using
`get_default_checkpointer`. The checkpointer saves both the last `k` checkpoints with the lowest
validation loss and the last training checkpoint.

The signature of the hook is:

```python
def get_default_checkpointer(
  outfolder: str,
  *,
  save_every_n_steps: int,
  max_to_keep: int = 5,
) -> tuple[Callable, Callable, Callable]
```

Its arguments are:
- `outfolder`: a folder specifying where to store the checkpoints.
- `save_every_n_steps`: after how many training steps to store a checkpoint.
- `max_to_keep`: the number of checkpoints to keep before starting to remove old checkpoints (to not clog the device).

For instance, you would construct the checkpointing function then like this:

```python
from blaxbird import get_default_checkpointer

hook_save, *_ = get_default_checkpointer(
  "checkpoints", save_every_n_steps=100
)
```

#### An EMA `hook`

We provide a hook for tracking an exponential moving average (EMA) of a
model's weights, constructed via `get_ema_hook`.

The signature is:

```python
def get_ema_hook(
  model: nnx.Module, decay: float = 0.999
) -> tuple[Callable, Callable]
```

Its arguments are:
- `model`: the model from which the EMA state is initialized.
- `decay`: the EMA decay rate.

It returns a tuple `(hook_fn, get_ema_model_fn)`:
- `hook_fn(step, *, model, **kwargs) -> None`: updates the tracked EMA
  weights every training step.
- `get_ema_model_fn(model: nnx.Module) -> nnx.Module`: returns a new,
  independent `nnx.Module` with the same structure as `model` but with the
  tracked EMA parameter values.

For instance, you would construct and use the EMA hook like this:

```python
from blaxbird import get_ema_hook

ema_hook, get_ema_model = get_ema_hook(model, decay=0.999)

train = train_fn(
  fns=(train_step, val_step),
  n_steps=n_steps,
  eval_every_n_steps=eval_every_n_steps,
  n_eval_batches=n_eval_batches,
  hooks=[ema_hook],
)
train(jr.key(1), optimizer, train_itr, val_itr)

ema_model = get_ema_model(optimizer.model)
```

Note: EMA state is not integrated with `get_default_checkpointer` --
saving and restoring EMA state alongside model checkpoints is not
covered here.

### Restoring a run

You can also use `get_default_checkpointer` to restart the run where you left off.
`get_default_checkpointer` in fact returns three functions, one for saving checkpoints and two for restoring
checkpoints:

```python
from blaxbird import get_default_checkpointer

save, restore_best, restore_last = get_default_checkpointer(
  "checkpoints", save_every_n_steps=100
)
```

You can then do either of:

```python
model = CNN(rngs=nnx.rnglib.Rngs(jr.key(1)))
optimizer = nnx.Optimizer(model, optax.adam(1e-4))

optimizer = restore_best(optimizer)
optimizer = restore_last(optimizer)
```

`restore_best`/`restore_last` take and return `optimizer` only -- the wrapped
model (`optimizer.model`) and `opt_state` are both updated in place on the
same optimizer instance, since `nnx.Optimizer` already owns the model it
wraps.

### Doing training

After having defined train functions, hooks and a mesh, you can train your model like this:

```python
train = train_fn(
  fns=(train_step, val_step),
  n_steps=n_steps,
  eval_every_n_steps=eval_every_n_steps,
  n_eval_batches=n_eval_batches,
  mesh=mesh,
  data_partition_spec=jax.sharding.PartitionSpec("data"),
  hooks=hooks,
  log_to_wandb=False,
)
train(jr.key(1), optimizer, train_itr, val_itr)
```

## Contributing

Contributions in the form of pull requests are more than welcome. A good way to
start is to check out issues labelled
[good first issue](https://github.com/dirmeier/surjectors/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

In order to contribute:

1) Clone `surjectors` and install `uv` from [here](https://docs.astral.sh/uv/getting-started/installation/).
2) Install all dependencies using `uv sync --all-groups`.
3) Install the Git hooks:

   ```bash
   uv run pre-commit install -t pre-commit -t commit-msg
   ```
4) Create a new branch locally, e.g. `git checkout -b feature/my-new-feature`.
5) Implement your contribution and ideally a test case.
6) Check your work (see below).
7) Submit a PR 🙂.

### Development commands

The project uses `uv` for everything:

```bash
uv sync --all-groups
uv run pytest
uv run ruff format blaxbird examples
uv run ruff check --fix blaxbird examples
uv run mypy blaxbird examples
uv run pre-commit run --all-files
```