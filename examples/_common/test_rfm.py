from _common.parameterizations import RFMConfig
from _common.rfm import rfm


def test_rfm_default_config_constructs_without_error():
  train_step, val_step, sample_fn = rfm(RFMConfig())
  assert callable(train_step)
  assert callable(val_step)
  assert callable(sample_fn)


def test_rfm_returns_objective_fns():
  from blaxbird._src._types import ObjectiveFns

  fns = rfm(RFMConfig())
  assert isinstance(fns, ObjectiveFns)
  assert fns.sample_fn is fns[2]
