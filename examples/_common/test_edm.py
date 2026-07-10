from _common.edm import edm
from _common.parameterizations import EDMConfig


def test_edm_default_config_constructs_without_error():
  train_step, val_step, sample_fn = edm(EDMConfig())
  assert callable(train_step)
  assert callable(val_step)
  assert callable(sample_fn)
