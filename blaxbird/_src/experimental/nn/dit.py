import jax
from einops import rearrange
from flax import nnx
from jax import numpy as jnp

from blaxbird._src.experimental.nn.embedding import timestep_embedding
from blaxbird._src.experimental.nn.mlp import MLP


def _modulate(inputs, shift, scale):  # noqa: ANN001, ANN202
  return inputs * (1.0 + scale[:, None]) + shift[:, None]


def get_sinusoidal_embedding_1d(length, embedding_dim):  # noqa: ANN001, ANN202
  emb = timestep_embedding(length.reshape(-1), embedding_dim)
  return emb


def sinusoidal_init(shape, dtype):  # noqa: ANN001, ANN202
  def get_sinusoidal_embedding_2d(grid, embedding_dim):  # noqa: ANN001, ANN202
    emb_h = get_sinusoidal_embedding_1d(grid[0], embedding_dim // 2)
    emb_w = get_sinusoidal_embedding_1d(grid[1], embedding_dim // 2)
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
  def __init__(  # noqa: PLR0913
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
  def __init__(  # noqa: PLR0913
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
  def __init__(  # noqa: PLR0913
    self,
    image_size: int,
    n_hidden_channels: int,
    patch_size: int,
    n_layers: int,
    n_heads: int,
    n_embedding_features=256,
    dropout_rate=0.0,
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
      rngs: random keys
    """
    self.image_size = image_size
    self.n_in_channels = image_size[-1]
    self.n_embedding_features = n_embedding_features
    self.patch_size = patch_size
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
      sinusoidal_init(
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
    self, inputs: jax.Array, times: jax.Array, context: jax.Array = None
  ):
    """Transform inputs through the DiT.

    Args:
      inputs: input in image form
      times: one-dimensional array
      context: conditioning variable in image form

    Returns:
      returns a jax
    """
    hidden = self._patchify(inputs)
    hidden = self._embed(hidden)
    times = self.time_embedding(
      timestep_embedding(times, self.n_embedding_features)
    )

    for block in self.dit_blocks:
      hidden = block(hidden, context=times)

    hidden = self.out_projection(hidden, times)
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
