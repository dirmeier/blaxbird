import jax.numpy as jnp

from _common.parameterizations import EDMParameterization


def test_denoise_composes_skip_and_out_scaling():
  params = EDMParameterization(sigma_data=0.5)
  inputs = jnp.ones((2, 4))
  sigma = jnp.array([1.0, 2.0])

  def fake_model(inputs, context, times):
    del context, times
    return inputs * 0.0 + 3.0  # constant model output

  out = params.denoise(fake_model, inputs, sigma, context=None)

  new_shape = (-1, 1)
  expected_skip = inputs * params.skip_scaling(sigma).reshape(new_shape)
  expected_out = 3.0 * params.out_scaling(sigma).reshape(new_shape)
  expected = expected_skip + expected_out
  assert jnp.allclose(out, expected)
