from typing import Callable

import jax
from flax import nnx


class MLP(nnx.Module):
  def __init__(
    self,
    in_features,
    output_features,
    *,
    kernel_init: nnx.initializers.Initializer | None = None,
    bias_init: nnx.initializers.Initializer | None = None,
    use_bias: bool = True,
    dropout_rate=None,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    activate_last: bool = False,
    rngs,
  ):
    features = [in_features] + output_features
    layers = []
    for index, (din, dout) in enumerate(zip(features[:-1], features[1:])):
      layers.append(
        nnx.Linear(
          in_features=din,
          out_features=dout,
          kernel_init=kernel_init,
          bias_init=bias_init,
          use_bias=use_bias,
          rngs=rngs,
        )
      )
    self.layers = tuple(layers)
    self.dropout_rate = dropout_rate
    self.activate_last = activate_last
    self.activation = activation
    if dropout_rate is not None:
      self.dropout_layer = nnx.Dropout(dropout_rate, rngs=rngs)

  def __call__(self, inputs):
    num_layers = len(self.layers)

    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < num_layers - 1 or self.activate_last:
        if self.dropout_rate is not None:
          out = self.dropout_layer(out)
        out = self.activation(out)
    return out
