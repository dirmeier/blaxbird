import jax
import pytest
from flax import nnx
from jax import random as jr
from jax.experimental import mesh_utils

from model import ShardedMLP


@pytest.mark.skipif(
  jax.local_device_count() < 4,
  reason="needs XLA_FLAGS=--xla_force_host_platform_device_count=4",
)
def test_sharded_mlp_splits_across_2d_mesh():
  mesh = jax.sharding.Mesh(
    mesh_utils.create_device_mesh((2, 2)), ("fsdp", "tp")
  )
  with mesh:
    model = ShardedMLP(8, 32, rngs=nnx.rnglib.Rngs(jr.key(0)))
    graphdef, state = nnx.split(model)
    sharding = nnx.get_named_sharding(state, mesh)
    state = jax.device_put(state, sharding)
    nnx.update(model, state)

    up_kernel = model.up.kernel.value
    assert up_kernel.shape == (8, 32)
    assert up_kernel.addressable_shards[0].data.shape == (4, 16)

    down_kernel = model.down.kernel.value
    assert down_kernel.shape == (32, 8)
    assert down_kernel.addressable_shards[0].data.shape == (16, 4)


def test_sharded_mlp_forward_pass_shape():
  """Runs on any device count -- proves the model works, not that it's
  actually sharded (see the skipif test above for that)."""
  import jax.numpy as jnp

  model = ShardedMLP(8, 32, rngs=nnx.rnglib.Rngs(jr.key(0)))
  out = model(jnp.ones((4, 8)))
  assert out.shape == (4, 8)
