"""Shared structural types for blaxbird's objective factories."""

from collections.abc import Callable
from typing import NamedTuple


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
