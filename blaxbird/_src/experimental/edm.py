import dataclasses

import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax import random as jr

from blaxbird._src.experimental import samplers


@dataclasses.dataclass
class EDMParameterization:
  n_sampling_steps: int = 25
  sigma_min: float = 0.002
  sigma_max: float = 80.0
  rho: float = 7.0
  sigma_data: float = 0.5
  P_mean: float = -1.2
  P_std: float = 1.2
  S_churn: float = 40
  S_min: float = 0.05
  S_max: float = 50
  S_noise: float = 1.003

  def sigma(self, eps):
    return jnp.exp(eps * self.P_std + self.P_mean)

  def loss_weight(self, sigma):
    return (jnp.square(sigma) + jnp.square(self.sigma_data)) / jnp.square(
      sigma * self.sigma_data
    )

  def skip_scaling(self, sigma):
    return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

  def out_scaling(self, sigma):
    return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

  def in_scaling(self, sigma):
    return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

  def noise_conditioning(self, sigma):
    return 0.25 * jnp.log(sigma)

  def sampling_sigmas(self, num_steps):
    rho_inv = 1 / self.rho
    step_idxs = jnp.arange(num_steps, dtype=jnp.float32)
    sigmas = (
      self.sigma_max**rho_inv
      + step_idxs
      / (num_steps - 1)
      * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
    ) ** self.rho
    return jnp.concatenate([sigmas, jnp.zeros_like(sigmas[:1])])

  def sigma_hat(self, sigma, num_steps):
    gamma = (
      jnp.minimum(self.S_churn / num_steps, 2**0.5 - 1)
      if self.S_min <= sigma <= self.S_max
      else 0
    )
    return sigma + gamma * sigma


@dataclasses.dataclass
class EDMConfig:
  n_sampling_steps: int = 25
  sampler: str = "heun"
  parameterization: EDMParameterization = dataclasses.field(
    default_factory=EDMParameterization
  )


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
