import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax import random as jr

from blaxbird._src.experimental import samplers
from blaxbird._src.experimental.parameterizations import EDMConfig


def edm(config: EDMConfig):
  """Construct denoising score-matching functions.

  Uses the EDM parameterization.

  Args:
    config: a EDMConfig object

  Returns:
    returns a tuple consisting of train_step, val_step and sampling functions
  """
  parameterization = config.parameterization

  def denoise(model, rng_key, inputs, sigma, context):
    new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
    inputs_t = inputs * parameterization.in_scaling(sigma).reshape(new_shape)
    noise_cond = parameterization.noise_conditioning(sigma)
    outputs = model(
      inputs=inputs_t,
      context=context,
      times=noise_cond,
    )
    skip = inputs * parameterization.skip_scaling(sigma).reshape(new_shape)
    outputs = outputs * parameterization.out_scaling(sigma).reshape(new_shape)
    outputs = skip + outputs
    return outputs

  def loss_fn(model, rng_key, batch):
    inputs = batch["inputs"]
    new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())

    epsilon_key, noise_key, rng_key = jr.split(rng_key, 3)
    epsilon = jr.normal(epsilon_key, (inputs.shape[0],))
    sigma = parameterization.sigma(epsilon)

    noise = jr.normal(noise_key, inputs.shape) * sigma.reshape(new_shape)
    denoise_key, rng_key = jr.split(rng_key)
    target_hat = denoise(
      model,
      denoise_key,
      inputs=inputs + noise,
      sigma=sigma,
      context=batch.get("context"),
    )

    loss = jnp.square(inputs - target_hat)
    loss = parameterization.loss_weight(sigma).reshape(new_shape) * loss
    return loss.mean()

  def train_step(model, rng_key, batch, **kwargs):
    return nnx.value_and_grad(loss_fn)(model, rng_key, batch)

  def val_step(model, rng_key, batch, **kwargs):
    return loss_fn(model, rng_key, batch)

  sampler = getattr(samplers, config.sampler + "sample_fn")(config)
  return train_step, val_step, sampler
