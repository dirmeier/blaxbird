# blaxbird [blækbɜːd]

[![ci](https://github.com/dirmeier/blaxbird/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/blaxbird/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/blaxbird.svg?colorB=black&style=flat)](https://pypi.org/project/blaxbird/)

> A high-level API to build and train NNX models

## About

`Blaxbird` [blækbɜːd] is a high-level API to easily build NNX models and train them on CPU or GPU.
Using `blaxbird` one can
- concisely define models and loss functions without the usual JAX/Flax verbosity,
- easily define checkpointers that save the best and most current network weights,
- distribute data and model weights over multiple processes or GPUs,
- define hooks that are periodically called during training.

## Example

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
train(jr.key(2), model, optimizer, train_itr, val_itr)
```

See a self-contained example in [examples/mnist_classification](examples/mnist_classification).

## Usage

`train_fn` is a higher order function with the following signature:

```python
def train_fn(
  *,
  fns: tuple[Callable, Callable],
  shardings: Optional[tuple[jax.NamedSharding, jax.NamedSharding]] = None,
  n_steps: int,
  eval_every_n_steps: int,
  n_eval_batches: int,
  log_to_wandb: bool = False,
  hooks: Iterable[Callable] = (),
) -> Callable:
  ...
```

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

### `shardings`

To specify how data and model weights are distributed over devices and processes,
`blaxbird` uses JAX' [sharding](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) functionality.

`shardings` is again specified by a tuple, one for the model sharding, the other for the data sharding.
An example is shown below, where we only distributed the data over `num_devices` devices.
You can, if you don't want to distribute anything, just set the argument to `None` or not specify it.

```python
def get_sharding():
  num_devices = jax.local_device_count()
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((num_devices,)), ("data",)
  )
  model_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
  data_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
  return model_sharding, data_sharding
```

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
hook =  hook_fn(metrics, val_iter, eval_every_n_steps)
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
  os.path.join(outfolder, "checkpoints"), save_every_n_steps=100
)
```

### Restoring a run

You can also use `get_default_checkpointer` to restart the run where you left off.
`get_default_checkpointer` in fact returns three functions, one for saving checkpoints and two for restoring
checkpoints:

```python
from blaxbird import get_default_checkpointer

save, restore_best, restore_last = get_default_checkpointer(
  os.path.join(outfolder, "checkpoints"), save_every_n_steps=100
)
```

You can then do either of:

```python
model = CNN(rngs=nnx.rnglib.Rngs(jr.key(1)))
optimizer = nnx.Optimizer(model, optax.adam(1e-4))

model, optimizer = restore_best(model, optimizer)
model, optimizer = restore_last(model, optimizer)
```

## Installation

To install the package from PyPI, call:

```bash
pip install blaxbird
```

To install the latest GitHub <RELEASE>, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/blaxbird@<RELEASE>
```

## Author

Simon Dirmeier <a href="mailto:simd@mailbox.org">simd@mailbox.org</a>
