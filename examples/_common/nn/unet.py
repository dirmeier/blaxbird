import jax
from flax import nnx
from jax import numpy as jnp


class ResBlock(nnx.Module):
  """Residual block with GroupNorm, SiLU, and AdaGN-style conditioning."""

  def __init__(
    self,
    in_channels,
    out_channels,
    n_embedding_features,
    *,
    dropout_rate=0.0,
    rngs,
  ):
    """Construct a residual block.

    Args:
      in_channels: number of input channels
      out_channels: number of output channels
      n_embedding_features: dimensionality of the conditioning embedding
        passed to __call__
      dropout_rate: float
      rngs: random keys
    """
    self.norm1 = nnx.GroupNorm(in_channels, num_groups=8, rngs=rngs)
    self.conv1 = nnx.Conv(
      in_channels, out_channels, (3, 3), padding="SAME", rngs=rngs
    )
    self.emb_proj = nnx.Linear(
      n_embedding_features, out_channels * 2, rngs=rngs
    )
    self.norm2 = nnx.GroupNorm(out_channels, num_groups=8, rngs=rngs)
    self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    self.conv2 = nnx.Conv(
      out_channels, out_channels, (3, 3), padding="SAME", rngs=rngs
    )
    self.skip = (
      None
      if in_channels == out_channels
      else nnx.Conv(in_channels, out_channels, (1, 1), rngs=rngs)
    )

  def __call__(self, inputs: jax.Array, embedding: jax.Array) -> jax.Array:
    """Transform inputs through the residual block.

    Args:
      inputs: input array, shape (batch, H, W, in_channels)
      embedding: conditioning embedding, shape (batch, n_embedding_features)

    Returns:
      returns a jax.Array, shape (batch, H, W, out_channels)
    """
    hidden = self.conv1(jax.nn.silu(self.norm1(inputs)))
    scale, shift = jnp.split(self.emb_proj(jax.nn.silu(embedding)), 2, axis=-1)
    hidden = self.norm2(hidden) * (1 + scale[:, None, None, :])
    hidden = hidden + shift[:, None, None, :]
    hidden = jax.nn.silu(hidden)
    hidden = self.dropout(hidden)
    hidden = self.conv2(hidden)
    skip = inputs if self.skip is None else self.skip(inputs)
    return skip + hidden


class AttentionBlock(nnx.Module):
  """Self-attention over spatial positions, with a residual connection."""

  def __init__(self, channels, n_heads, *, rngs):
    """Construct an attention block.

    Args:
      channels: number of channels (must be divisible by n_heads)
      n_heads: number of attention heads
      rngs: random keys
    """
    self.norm = nnx.GroupNorm(channels, num_groups=8, rngs=rngs)
    self.attn = nnx.MultiHeadAttention(
      num_heads=n_heads, in_features=channels, rngs=rngs, decode=False
    )

  def __call__(self, inputs: jax.Array) -> jax.Array:
    """Apply self-attention across the H*W spatial positions.

    Args:
      inputs: input array, shape (batch, H, W, channels)

    Returns:
      returns a jax.Array, same shape as inputs
    """
    b, h, w, c = inputs.shape
    hidden = self.norm(inputs)
    hidden = hidden.reshape(b, h * w, c)
    hidden = self.attn(hidden)
    hidden = hidden.reshape(b, h, w, c)
    return inputs + hidden


class Downsample(nnx.Module):
  """Halve spatial resolution via a stride-2 convolution."""

  def __init__(self, channels, *, rngs):
    """Construct a downsampling block.

    Args:
      channels: number of channels (unchanged by downsampling)
      rngs: random keys
    """
    self.conv = nnx.Conv(
      channels, channels, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs
    )

  def __call__(self, inputs: jax.Array) -> jax.Array:
    """Downsample inputs by 2x.

    Args:
      inputs: input array, shape (batch, H, W, channels)

    Returns:
      returns a jax.Array, shape (batch, H // 2, W // 2, channels)
    """
    return self.conv(inputs)


class Upsample(nnx.Module):
  """Double spatial resolution via nearest-neighbor resize + convolution."""

  def __init__(self, channels, *, rngs):
    """Construct an upsampling block.

    Args:
      channels: number of channels (unchanged by upsampling)
      rngs: random keys
    """
    self.conv = nnx.Conv(channels, channels, (3, 3), padding="SAME", rngs=rngs)

  def __call__(self, inputs: jax.Array) -> jax.Array:
    """Upsample inputs by 2x.

    Args:
      inputs: input array, shape (batch, H, W, channels)

    Returns:
      returns a jax.Array, shape (batch, H * 2, W * 2, channels)
    """
    b, h, w, c = inputs.shape
    resized = jax.image.resize(inputs, (b, h * 2, w * 2, c), method="nearest")
    return self.conv(resized)
