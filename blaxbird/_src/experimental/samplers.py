import dataclasses
from collections.abc import Callable

import chex
import numpy as np
from jax import numpy as jnp
from jax import random as jr


def euler_sample_fn(config):
  def sample_fn(model, rng_key, sample_shape=(), *, context=None):
    if context is not None:
      chex.assert_equal(sample_shape[0], len(context))
    dt = 1.0 / config.n_sampling_steps
    samples = jr.normal(rng_key, sample_shape)
    for i in range(config.n_sampling_steps):
      times = i / config.n_sampling_steps
      times = times * (config.time_max - config.time_eps) + config.time_eps
      times = jnp.repeat(times, samples.shape[0])
      vt = model(inputs=samples, times=times, context=context)
      samples = samples + vt * dt
    return samples

  return sample_fn


def heun_sampler_fn(config):
  def denoise(model, rng_key, inputs, sigma, context):
    new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
    inputs_t = inputs * config.in_scaling(sigma).reshape(new_shape)
    noise_cond = config.noise_conditioning(sigma)
    outputs = model(
      inputs=inputs_t,
      context=context,
      times=noise_cond,
    )
    skip = inputs * config.skip_scaling(sigma).reshape(new_shape)
    outputs = outputs * config.out_scaling(sigma).reshape(new_shape)
    outputs = skip + outputs
    return outputs

  def sample_fn(model, rng_key, sample_shape=(), *, context=None):
    if context is not None:
      chex.assert_equal(sample_shape[0], len(context))
    n = context.shape[0]
    noise_key, rng_key = jr.split(rng_key)
    sigmas = config.sampling_sigmas(config.n_sampling_steps)
    samples = jr.normal(rng_key, sample_shape) * sigmas[0]

    for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
      pred_key1, pred_key2, rng_key = jr.split(rng_key, 3)
      sample_curr = samples
      pred_curr = denoise(
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
        pred_next = denoise(
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
