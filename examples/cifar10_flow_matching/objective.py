"""Rectified flow matching training objective for the CIFAR-10 DiT."""

import dataclasses

import chex
import numpy as np
from typing import NamedTuple, Callable

from flax import nnx
from jax import numpy as jnp
from jax import random as jr

__all__ = [
  "RFMParameterization",
  "RFMConfig",
  "rfm"
]

class ObjectiveFns(NamedTuple):
  """Functions returned by an objective factory (e.g. edm(), rfm()).

  Attributes:
    train_step: gradient-step function with signature
      (model, rng_key, batch, **kwargs) -> (loss, grads).
    val_step: validation function with signature
      (model, rng_key, batch, **kwargs) -> loss.
    sample_fn: sampling function with signature
      (model, rng_key, sample_shape, *, context=None) -> samples.
  """

  train_step: Callable
  val_step: Callable
  sample_fn: Callable


@dataclasses.dataclass
class RFMParameterization:
  t_eps: float = 1e-5
  t_max: float = 1.0

  def sampling_sigmas(self, num_steps):
    return jnp.linspace(self.t_eps, self.t_max, num_steps)


@dataclasses.dataclass
class RFMConfig:
  n_sampling_steps: int = 25
  parameterization: RFMParameterization = dataclasses.field(
    default_factory=RFMParameterization
  )


def _forward_process(inputs, times, noise):
  new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
  times = times.reshape(new_shape)
  inputs_t = times * inputs + (1.0 - times) * noise
  return inputs_t


def _euler_sample_fn(config: RFMConfig):
  def sample_fn(model, rng_key, sample_shape=(), *, context=None):
    if context is not None:
      chex.assert_equal(sample_shape[0], len(context))
    dt = 1.0 / config.n_sampling_steps
    samples = jr.normal(rng_key, sample_shape)
    time_steps = config.parameterization.sampling_sigmas(
      config.n_sampling_steps
    )
    for times in time_steps:
      times = jnp.repeat(times, samples.shape[0])  # noqa: PLW2901
      vt = model(inputs=samples, times=times, context=context)
      samples = samples + vt * dt
    return samples

  return sample_fn


def rfm(config: RFMConfig = RFMConfig()) -> ObjectiveFns:
  """Construct rectified flow matching train/val/sample functions.

  Args:
    config: an RFMConfig object

  Returns:
    an ObjectiveFns with train_step, val_step and an Euler sample_fn
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
    del kwargs
    return nnx.value_and_grad(_loss_fn)(model, rng_key, batch)

  def val_step(model, rng_key, batch, **kwargs):
    del kwargs
    return _loss_fn(model, rng_key, batch)

  return ObjectiveFns(
    train_step=train_step,
    val_step=val_step,
    sample_fn=_euler_sample_fn(config),
  )
