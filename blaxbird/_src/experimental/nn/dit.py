import numpy as np
from einops import rearrange
from flax import nnx
from jax import numpy as jnp

from blaxbird._src.experimental.nn.embedding import timestep_embedding
from blaxbird._src.experimental.nn.mlp import MLP


def _modulate(inputs, shift, scale):
  return inputs * (1.0 + scale[:, None]) + shift[:, None]


class OutProjection(nnx.Module):
  def __init__(self, hidden_size, patch_size, out_channels, *, rngs):
    super().__init__()
    self.norm = nnx.LayerNorm(hidden_size, rngs=rngs)
    self.out = nnx.Linear(
      hidden_size, patch_size * patch_size * out_channels, rngs=rngs
    )
    self.ada = nnx.Sequential(
      nnx.swish, nnx.Linear(hidden_size, 2 * hidden_size, rngs=rngs)
    )

  def __call__(self, inputs, context):
    shift, scale = jnp.split(self.ada(context), 2, -1)
    outs = self.out(_modulate(self.norm(inputs), shift, scale))
    return outs


class DiTBlock(nnx.Module):
  def __init__(
    self,
    hidden_size: int,
    n_heads: int,
    dropout_rate: float = 0.1,
    *,
    rngs: nnx.rnglib.Rngs,
  ):
    super().__init__()
    self.ada = nnx.Linear(hidden_size, hidden_size * 6, rngs=rngs)
    self.layer_norm1 = nnx.LayerNorm(
      hidden_size, use_scale=False, use_bias=False, rngs=rngs
    )
    self.self_attn = nnx.MultiHeadAttention(
      num_heads=n_heads, in_features=hidden_size, rngs=rngs
    )
    self.layer_norm2 = nnx.LayerNorm(
      hidden_size, use_scale=False, use_bias=False, rngs=rngs
    )
    self.mlp = MLP(
      hidden_size,
      [hidden_size * 4, hidden_size],
      dropout_rate=dropout_rate,
      rngs=rngs,
    )

  def __call__(self, inputs, context, **kwargs):
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
    n_in_channels,
    n_hidden_channels,
    patch_size,
    n_layers,
    n_heads,
    n_embedding_features=256,
    dropout_rate=0.0,
    *,
    rngs,
  ):
    self.n_in_channels = n_in_channels
    self.n_embedding_features = n_embedding_features
    self.patch_size = patch_size
    self.time_embedding = nnx.Sequential(
      nnx.Linear(n_embedding_features, n_embedding_features, rngs=rngs),
      nnx.swish,
      nnx.Linear(n_embedding_features, n_embedding_features, rngs=rngs),
      nnx.swish,
    )
    self.patchify = nnx.Conv(
      n_in_channels,
      n_hidden_channels,
      (patch_size, patch_size),
      (patch_size, patch_size),
      padding="VALID",
      kernel_init=nnx.initializers.xavier_uniform(),
      rngs=rngs,
    )
    self.patch_embedding = nnx.Param(
      nnx.initializers.lecun_normal()(
        rngs.params(), (1, n_hidden_channels, n_hidden_channels)
      )
    )
    self.dit_blocks = tuple(
      [
        DiTBlock(n_hidden_channels, n_heads, dropout_rate, rngs=rngs)
        for _ in n_layers
      ]
    )
    self.final_time_embedding = nnx.Linear(
      n_hidden_channels, n_hidden_channels * 2, rngs=rngs
    )
    self.out_projection = OutProjection(
      n_hidden_channels, patch_size, n_in_channels, rngs=rngs
    )

  def _patchify(self, inputs):
    B, H, W, C = inputs.shape
    n_patches = H // self.patch_size
    hidden = self.patchify(inputs)
    outputs = rearrange(
      hidden, "b h w c -> b (h w) c", h=n_patches, w=n_patches
    )
    return outputs

  def _unpatchify(self, inputs):
    B, HW, *_ = inputs.shape
    h = w = int(np.sqrt(HW))
    p = q = self.patch_size
    hidden = jnp.reshape(inputs, (B, h, w, p, q, self.n_in_channels))
    outputs = rearrange(
      hidden, "b h w p q c -> b (h p) (w q) c", h=h, w=w, p=q, q=q
    )
    return outputs

  def _embed(self, inputs):
    return inputs + self.patch_embedding.value

  def __call__(self, inputs, times):
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


def SmallDiT(patch_size=2, **kwargs):
  return DiT(
    n_hidden_channels=384,
    patch_size=patch_size,
    n_layers=12,
    n_heads=6,
    **kwargs,
  )


def BaseDiT(patch_size=2, **kwargs):
  return DiT(
    n_hidden_channels=768,
    patch_size=patch_size,
    n_layers=12,
    n_heads=12,
    **kwargs,
  )


def LargeDiT(patch_size=2, **kwargs):
  return DiT(
    n_hidden_channels=1024,
    patch_size=patch_size,
    n_layers=24,
    n_heads=16,
    **kwargs,
  )


def XtraLargeDiT(patch_size=2, **kwargs):
  return DiT(
    n_hidden_channels=1152,
    patch_size=patch_size,
    n_layers=28,
    n_heads=16,
    **kwargs,
  )





