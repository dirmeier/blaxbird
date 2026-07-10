from blaxbird._src._types import ObjectiveFns


def test_objective_fns_fields_are_named():
  def train_step():
    pass

  def val_step():
    pass

  def sample_fn():
    pass

  fns = ObjectiveFns(
    train_step=train_step, val_step=val_step, sample_fn=sample_fn
  )
  assert fns.train_step is train_step
  assert fns.val_step is val_step
  assert fns.sample_fn is sample_fn
  # still unpacks positionally like a plain tuple
  a, b, c = fns
  assert (a, b, c) == (train_step, val_step, sample_fn)
