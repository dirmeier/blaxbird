import pytest

from _common import samplers


def test_get_sampler_fn_returns_euler():
  fn = samplers.get_sampler_fn("euler")
  assert fn is samplers.euler_sample_fn


def test_get_sampler_fn_returns_heun():
  fn = samplers.get_sampler_fn("heun")
  assert fn is samplers.heun_sample_fn


def test_get_sampler_fn_unknown_name_raises():
  with pytest.raises(ValueError, match="unknown sampler"):
    samplers.get_sampler_fn("not-a-sampler")
