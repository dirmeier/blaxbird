import dataclasses

import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax import random as jr

from blaxbird._src.experimental import samplers


def _forward_process(inputs, times, noise):
  new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
  times = times.reshape(new_shape)
  inputs_t = times * inputs + (1.0 - times) * noise
  return inputs_t


@dataclasses.dataclass
class FlowMatchingConfig:
  n_sampling_steps: int = 25
  time_eps: float = 1e-3
  time_max: float = 1.0
  sampler: str = "euler"


def flow_matching(config: FlowMatchingConfig):
  def loss_fn(model, rng_key, batch):
    inputs = batch["inputs"]
    time_key, rng_key = jr.split(rng_key)
    times = jr.uniform(time_key, shape=(inputs.shape[0],))
    times = times * (config.time_max - config.time_eps) + config.time_eps
    noise_key, rng_key = jr.split(rng_key)
    noise = jr.normal(noise_key, inputs.shape)
    inputs_t = _forward_process(inputs, times, noise)
    vs = model(
      inputs=inputs_t, times=times * 999.0, context=batch.get("context")
    )
    target = inputs - noise
    loss = jnp.mean(jnp.square(target - vs))
    return loss

  def train_step(model, rng_key, batch, **kwargs):
    return nnx.value_and_grad(loss_fn)(model, rng_key, batch)

  def val_step(model, rng_key, batch, **kwargs):
    return loss_fn(model, rng_key, batch)

  sampler = getattr(samplers, config.sampler + "sample_fn")(config)
  return train_step, val_step, sampler
