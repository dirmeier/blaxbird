from collections.abc import Callable

import chex
import jax
from flax import nnx
from jax import numpy as jnp
from jax import random as jr

from _common.parameterizations import EDMConfig, RFMConfig


def euler_sample_fn(config: RFMConfig):
  """Construct an Euler sampler for flow matching.

  Args:
    config: a FlowMatchingConfig object

  Returns:
    returns a callable that can be used to sample from a flow matching model
  """

  def sample_fn(
    model: nnx.Module,
    rng_key: jax.Array,
    sample_shape: tuple = (),
    *,
    context: jax.Array = None,
  ) -> jax.Array:
    """Sample from a flow matching model.

    Args:
      model: a nnx.Module that is used as the learned vector field in flow
       matching
      rng_key: a jax.random.key object
      sample_shape: the shape of the data to be generated, where the first axis
        is the batch dimension and the other axes are the feature dimensions
      context: a conditioning variable (if used)

    Returns:
      returns a sample from the model
    """
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


def heun_sample_fn(config: EDMConfig):
  """Construct a Heun sampler for denoising score matching.

  Args:
    config: a EDMConfig object

  Returns:
    returns a callable that can be used to sample from a score matching model
  """
  params = config.parameterization

  def sample_fn(
    model: nnx.Module,
    rng_key: jax.Array,
    sample_shape: tuple = (),
    *,
    context: jax.Array = None,
  ) -> jax.Array:
    """Sample from a score matching model.

    Args:
      model: a nnx.Module that is used as the learned score model in score
        matching
      rng_key: a jax.random.key object
      sample_shape: the shape of the data to be generated, where the first axis
        is the batch dimension and the other axes are the feature dimensions
      context: a conditioning variable (if used)

    Returns:
      returns a sample from the model
    """
    if context is not None:
      chex.assert_equal(sample_shape[0], len(context))
    n = context.shape[0]
    noise_key, rng_key = jr.split(rng_key)
    sigmas = params.sampling_sigmas(config.n_sampling_steps)
    samples = jr.normal(rng_key, sample_shape) * sigmas[0]

    for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
      sample_curr = samples
      pred_curr = params.denoise(
        model,
        inputs=sample_curr,
        sigma=jnp.repeat(sigma, n),
        context=context,
      )
      d_cur = (sample_curr - pred_curr) / sigma
      samples = sample_curr + d_cur * (sigma_next - sigma)
      # second order correction
      if i < config.n_sampling_steps - 1:
        pred_next = params.denoise(
          model,
          inputs=samples,
          sigma=jnp.repeat(sigma_next, n),
          context=context,
        )
        d_prime = (samples - pred_next) / sigma_next
        samples = sample_curr + (sigma_next - sigma) * (
          0.5 * d_cur + 0.5 * d_prime
        )
    return samples

  return sample_fn


SAMPLERS: dict[str, Callable] = {
  "euler": euler_sample_fn,
  "heun": heun_sample_fn,
}


def get_sampler_fn(name: str) -> Callable:
  """Look up a sampler constructor by name.

  Args:
    name: one of the registered sampler names, currently "euler" or "heun".

  Returns:
    the sampler-constructor callable registered under `name`. Calling it
    with a config object returns the actual `sample_fn`.

  Raises:
    ValueError: if `name` is not a registered sampler.
  """
  if name not in SAMPLERS:
    raise ValueError(
      f"unknown sampler {name!r}, expected one of {sorted(SAMPLERS)}"
    )
  return SAMPLERS[name]
