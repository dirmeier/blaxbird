from collections.abc import Callable

import jax
from einops import rearrange
from flax import nnx
from jax import numpy as jnp


def timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
  half = embedding_dim // 2
  freqs = jnp.exp(-jnp.log(10_000) * jnp.arange(0, half) / half)
  emb = timesteps.astype(dtype)[:, None] * freqs[None, ...]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  return emb


class MLP(nnx.Module):
  def __init__(
    self,
    in_features: int,
    output_features: tuple[int, ...],
    *,
    kernel_init: nnx.initializers.Initializer = nnx.initializers.lecun_normal(),
    bias_init: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
    use_bias: bool = True,
    dropout_rate: float | None = None,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    activate_last: bool = False,
    rngs: nnx.rnglib.Rngs,
  ):
    features = [in_features] + list(output_features)
    layers = []
    for din, dout in zip(features[:-1], features[1:], strict=True):
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

  def __call__(self, inputs: jax.Array):
    """Project inputs through the MLP.

    Args:
      inputs: jax.Array

    Returns:
      jax.Array
    """
    num_layers = len(self.layers)

    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < num_layers - 1 or self.activate_last:
        if self.dropout_rate is not None:
          out = self.dropout_layer(out)
        out = self.activation(out)
    return out


def _modulate(inputs, shift, scale):
  return inputs * (1.0 + scale[:, None]) + shift[:, None]


def _get_sinusoidal_embedding_1d(length, embedding_dim):
  return timestep_embedding(length.reshape(-1), embedding_dim)


def _sinusoidal_init(shape, dtype):
  del dtype

  def get_sinusoidal_embedding_2d(grid, embedding_dim):
    emb_h = _get_sinusoidal_embedding_1d(grid[0], embedding_dim // 2)
    emb_w = _get_sinusoidal_embedding_1d(grid[1], embedding_dim // 2)
    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb

  _, n_h_patches, n_w_patches, embedding_dim = shape
  grid_h = jnp.arange(n_h_patches, dtype=jnp.float32)
  grid_w = jnp.arange(n_w_patches, dtype=jnp.float32)
  grid = jnp.meshgrid(grid_w, grid_h)

  grid = jnp.stack(grid, axis=0)
  grid = grid.reshape([2, 1, n_w_patches, n_h_patches])
  pos_embed = get_sinusoidal_embedding_2d(grid, embedding_dim)

  return jnp.expand_dims(pos_embed, 0)  # (1, H*W, D)


class OutProjection(nnx.Module):
  def __init__(
    self, hidden_size, n_embedding_features, patch_size, out_channels, *, rngs
  ):
    super().__init__()
    self.ada = nnx.Sequential(
      nnx.silu, nnx.Linear(n_embedding_features, 2 * hidden_size, rngs=rngs)
    )
    self.norm = nnx.LayerNorm(hidden_size, rngs=rngs)
    self.out = nnx.Linear(
      hidden_size, patch_size * patch_size * out_channels, rngs=rngs
    )

  def __call__(self, inputs, context):
    shift, scale = jnp.split(self.ada(context), 2, -1)
    outs = self.out(_modulate(self.norm(inputs), shift, scale))
    return outs


class DiTBlock(nnx.Module):
  def __init__(
    self,
    hidden_size: int,
    n_embedding_features: int,
    *,
    n_heads: int,
    dropout_rate: float = 0.1,
    rngs: nnx.rnglib.Rngs,
  ):
    """Diffusion-Transformer block.

    Args:
      hidden_size: number of features of the hidden layers
      n_embedding_features: number o features of time embedding
      n_heads: number of transformer heads
      dropout_rate: float
      rngs: random keys
    """
    super().__init__()
    self.ada = nnx.Sequential(
      nnx.silu, nnx.Linear(n_embedding_features, hidden_size * 6, rngs=rngs)
    )

    self.layer_norm1 = nnx.LayerNorm(
      hidden_size, use_scale=False, use_bias=False, rngs=rngs
    )
    self.self_attn = nnx.MultiHeadAttention(
      num_heads=n_heads, in_features=hidden_size, rngs=rngs, decode=False
    )
    self.layer_norm2 = nnx.LayerNorm(
      hidden_size, use_scale=False, use_bias=False, rngs=rngs
    )
    self.mlp = MLP(
      hidden_size,
      (hidden_size * 4, hidden_size),
      dropout_rate=dropout_rate,
      rngs=rngs,
    )

  def __call__(self, inputs: jax.Array, context: jax.Array) -> jax.Array:
    """Transform inputs through the DiT block.

    Args:
      inputs: input array
      context: values to condition on

    Returns:
      returns a jax.Array
    """
    hidden = inputs
    adaln_norm = self.ada(context)
    attn, gate = jnp.split(adaln_norm, 2, axis=-1)

    pre_shift, pre_scale, post_scale = jnp.split(attn, 3, -1)
    intermediate = _modulate(self.layer_norm1(hidden), pre_shift, pre_scale)
    intermediate = self.self_attn(intermediate)
    hidden = hidden + post_scale[:, None] * intermediate

    pre_shift, pre_scale, post_scale = jnp.split(gate, 3, -1)
    intermediate = _modulate(self.layer_norm2(hidden), pre_shift, pre_scale)
    intermediate = self.mlp(intermediate)
    outputs = hidden + post_scale[:, None] * intermediate

    return outputs


class DiT(nnx.Module):
  def __init__(
    self,
    image_size: tuple[int, int, int],
    n_hidden_channels: int,
    patch_size: int,
    n_layers: int,
    n_heads: int,
    n_embedding_features=256,
    dropout_rate=0.0,
    n_classes: int | None = None,
    *,
    rngs: nnx.rnglib.Rngs,
  ):
    """Diffusion-Transformer.

    Args:
      image_size: size of the image, e.g., (32, 32, 3)
      n_hidden_channels: number if hidden channels
      patch_size: size of each path
      n_layers: integer
      n_heads: integer
      n_embedding_features: integer
      dropout_rate: float
      n_classes: number of classes to condition on, or None for an
        unconditional model. When set, __call__ requires a `context`
        argument of integer class labels; when None, __call__ requires
        `context=None`.
      rngs: random keys
    """
    self.image_size = image_size
    self.n_in_channels = image_size[-1]
    self.n_embedding_features = n_embedding_features
    self.patch_size = patch_size
    self.n_classes = n_classes
    if n_classes is not None:
      self.class_embedding = nnx.Embed(
        n_classes, n_embedding_features, rngs=rngs
      )
    self.time_embedding = nnx.Sequential(
      nnx.Linear(n_embedding_features, n_embedding_features, rngs=rngs),
      nnx.swish,
      nnx.Linear(n_embedding_features, n_embedding_features, rngs=rngs),
      nnx.swish,
    )
    self.patchify = nnx.Conv(
      self.n_in_channels,
      n_hidden_channels,
      (patch_size, patch_size),
      (patch_size, patch_size),
      padding="VALID",
      kernel_init=nnx.initializers.xavier_uniform(),
      rngs=rngs,
    )
    self.patch_embedding = nnx.Param(
      _sinusoidal_init(
        (
          1,
          image_size[0] // patch_size,
          image_size[1] // patch_size,
          n_hidden_channels,
        ),
        None,
      ),
    )
    self.dit_blocks = tuple(
      [
        DiTBlock(
          n_hidden_channels,
          n_embedding_features,
          n_heads=n_heads,
          dropout_rate=dropout_rate,
          rngs=rngs,
        )
        for _ in range(n_layers)
      ]
    )
    self.out_projection = OutProjection(
      n_hidden_channels,
      n_embedding_features,
      patch_size,
      self.n_in_channels,
      rngs=rngs,
    )

  def _patchify(self, inputs):
    n_h_patches = self.image_size[0] // self.patch_size
    n_w_patches = self.image_size[1] // self.patch_size
    hidden = self.patchify(inputs)
    outputs = rearrange(
      hidden, "b h w c -> b (h w) c", h=n_h_patches, w=n_w_patches
    )
    return outputs

  def _unpatchify(self, inputs):
    H = self.image_size[0] // self.patch_size
    W = self.image_size[1] // self.patch_size
    P = Q = self.patch_size
    hidden = jnp.reshape(inputs, (-1, H, W, P, Q, self.n_in_channels))
    outputs = rearrange(
      hidden, "b h w p q c -> b (h p) (w q) c", h=H, w=W, p=P, q=Q
    )
    return outputs

  def _embed(self, inputs):
    return inputs + jax.lax.stop_gradient(self.patch_embedding.value)

  def __call__(
    self,
    inputs: jax.Array,
    times: jax.Array,
    context: jax.Array | None = None,
  ):
    """Transform inputs through the DiT.

    Args:
      inputs: input in image form
      times: one-dimensional array
      context: integer class labels, shape (batch,), required if this DiT
        was constructed with n_classes set; must be None otherwise.

    Returns:
      returns a jax.Array, same shape as `inputs`

    Raises:
      ValueError: if `context` is None but n_classes was set, or if
        `context` is given but n_classes was not set.
    """
    if self.n_classes is not None and context is None:
      raise ValueError(
        "this DiT was constructed with n_classes set, so context "
        "(integer class labels) must be provided"
      )
    if self.n_classes is None and context is not None:
      raise ValueError(
        "this DiT was constructed without n_classes, so context must "
        "be None -- pass n_classes at construction to condition on it"
      )

    hidden = self._patchify(inputs)
    hidden = self._embed(hidden)
    embedding = self.time_embedding(
      timestep_embedding(times, self.n_embedding_features)
    )
    if context is not None:
      embedding = embedding + self.class_embedding(context)

    for block in self.dit_blocks:
      hidden = block(hidden, context=embedding)

    hidden = self.out_projection(hidden, embedding)
    outputs = self._unpatchify(hidden)
    return outputs


def SmallDiT(image_size, patch_size=2, **kwargs):
  return DiT(
    image_size,
    n_hidden_channels=384,
    patch_size=patch_size,
    n_layers=12,
    n_heads=6,
    **kwargs,
  )


def BaseDiT(image_size, patch_size=2, **kwargs):
  return DiT(
    image_size,
    n_hidden_channels=768,
    patch_size=patch_size,
    n_layers=12,
    n_heads=12,
    **kwargs,
  )


def LargeDiT(image_size, patch_size=2, **kwargs):
  return DiT(
    image_size,
    n_hidden_channels=1024,
    patch_size=patch_size,
    n_layers=24,
    n_heads=16,
    **kwargs,
  )


def XtraLargeDiT(image_size, patch_size=2, **kwargs):
  return DiT(
    image_size,
    n_hidden_channels=1152,
    patch_size=patch_size,
    n_layers=28,
    n_heads=16,
    **kwargs,
  )
