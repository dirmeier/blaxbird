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
class RFMParameterization:
  t_eps: float = 1e-5
  t_max: float = 1.0

  def sigma(self, eps):
    return self.t_eps + (self.t_max - self.t_eps)

  def loss_weight(self, t):
    return 1.0

  def skip_scaling(self, t):
    return 0.0

  def out_scaling(self, t):
    return 1.0

  def in_scaling(self, t):
    return 1.0

  def noise_conditioning(self, t):
    return t

  def sampling_sigmas(self, num_steps):
    return jnp.linspace(self.t_eps, self.t_max, num_steps)

  def sigma_hat(self, t, num_steps):
    return t


@dataclasses.dataclass
class RFMConfig:
  n_sampling_steps: int = 25
  sampler: str = "euler"
  parameterization: RFMParameterization = dataclasses.field(
    default_factory=RFMParameterization
  )


def rfm(config: RFMConfig = RFMConfig()):
  """Construct rectified flow matching functions.

  Args:
    config: a FlowMatchingConfig object

  Returns:
    returns a tuple consisting of train_step, val_step and sampling functions
  """
  parameterization = config.parameterization

  def _loss_fn(model, rng_key, batch):
    inputs = batch["inputs"]
    time_key, rng_key = jr.split(rng_key)
    times = jr.uniform(time_key, shape=(inputs.shape[0],))
    times = (
      times * (parameterization.t_max - parameterization.t_eps)
      + parameterization.t_eps
    )
    noise_key, rng_key = jr.split(rng_key)
    noise = jr.normal(noise_key, inputs.shape)
    inputs_t = _forward_process(inputs, times, noise)
    vt = model(inputs=inputs_t, times=times, context=batch.get("context"))
    ut = inputs - noise
    loss = jnp.mean(jnp.square(ut - vt))
    return loss

  def train_step(model, rng_key, batch, **kwargs):
    return nnx.value_and_grad(_loss_fn)(model, rng_key, batch)

  def val_step(model, rng_key, batch, **kwargs):
    return _loss_fn(model, rng_key, batch)

  sampler = getattr(samplers, config.sampler + "_sample_fn")(config)
  return train_step, val_step, sampler
