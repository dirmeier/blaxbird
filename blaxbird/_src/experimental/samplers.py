import chex
import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax import random as jr

from blaxbird._src.experimental.edm import EDMConfig
from blaxbird._src.experimental.rfm import (
  RFMConfig,
)


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


def heun_sampler_fn(config: EDMConfig):
  """Construct a Heun sampler for denoising score matching.

  Args:
    config: a EDMConfig object

  Returns:
    returns a callable that can be used to sample from a score matching model
  """
  params = config.parameterization

  # ruff: noqa: ANN001, ANN202, ANN003
  def _denoise(model, rng_key, inputs, sigma, context):
    new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
    inputs_t = inputs * params.in_scaling(sigma).reshape(new_shape)
    noise_cond = params.noise_conditioning(sigma)
    outputs = model(
      inputs=inputs_t,
      context=context,
      times=noise_cond,
    )
    skip = inputs * params.skip_scaling(sigma).reshape(new_shape)
    outputs = outputs * params.out_scaling(sigma).reshape(new_shape)
    outputs = skip + outputs
    return outputs

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
      pred_key1, pred_key2, rng_key = jr.split(rng_key, 3)
      sample_curr = samples
      pred_curr = _denoise(
        model,
        pred_key1,
        inputs=sample_curr,
        sigma=jnp.repeat(sigma, n),
        context=context,
      )
      d_cur = (sample_curr - pred_curr) / sigma
      samples = sample_curr + d_cur * (sigma_next - sigma)
      # second order correction
      if i < config.n_sampling_steps - 1:
        pred_next = _denoise(
          model,
          pred_key2,
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
