# blaxbird [blækbɜːd]

## About

A high-level API to build and train NNX models.

Define the module
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


def train_step(model, rng_key, batch, **kwargs):
    return nnx.value_and_grad(loss_fn)(model, batch["image"], batch["label"])


def val_step(model, rng_key, batch, **kwargs):
    return loss_fn(model, batch["image"], batch["label"])
```

Define the trainer
```python
from jax import random as jr
from flax import nnx

from blaxbird import train_fn

model = CNN(rngs=nnx.rnglib.Rngs(jr.key(1)))
optimizer = get_optimizer(model)

train = train_fn(
  fns=(train_step, val_step),
  n_steps=n_steps,
  n_eval_frequency=n_eval_frequency,
  n_eval_batches=n_eval_batches,
)
train(jr.key(2), model, optimizer, train_itr, val_itr)
```

## Author

Simon Dirmeier <a href="mailto:simd@mailbox.org">simd@mailbox.org</a>
