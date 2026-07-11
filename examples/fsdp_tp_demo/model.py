"""A tiny MLP with explicit FSDP+TP sharding annotations, demonstrating
flax.nnx's with_partitioning / get_named_sharding mechanism on a 2D
device mesh. Verified live (see the plan doc) with a simulated 4-device
mesh reshaped to (2, 2): the up-projection kernel shards ("fsdp", "tp"),
the down-projection kernel shards ("tp", "fsdp") -- standard
Megatron-style column-parallel-then-row-parallel MLP sharding, combined
with FSDP on the same two axes.
"""

import jax
from flax import nnx


class ShardedMLP(nnx.Module):
  """Two-layer MLP with FSDP+TP-annotated kernels."""

  def __init__(self, d_model, d_ff, *, rngs):
    """Construct a sharded MLP.

    Args:
      d_model: input/output dimensionality
      d_ff: hidden (expansion) dimensionality
      rngs: random keys
    """
    self.up = nnx.Linear(
      d_model,
      d_ff,
      rngs=rngs,
      kernel_init=nnx.with_partitioning(
        nnx.initializers.lecun_normal(), ("fsdp", "tp")
      ),
    )
    self.down = nnx.Linear(
      d_ff,
      d_model,
      rngs=rngs,
      kernel_init=nnx.with_partitioning(
        nnx.initializers.lecun_normal(), ("tp", "fsdp")
      ),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Apply the MLP.

    Args:
      x: input array, shape (batch, d_model)

    Returns:
      jax.Array, shape (batch, d_model)
    """
    return self.down(jax.nn.relu(self.up(x)))
