import jax
from flax import nnx
from jax import numpy as jnp

from _common.nn.embedding import timestep_embedding


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


class UNet(nnx.Module):
  """ADM/EDM-style UNet: ResBlocks, attention, skip connections."""

  def __init__(  # noqa: PLR0913
    self,
    image_size,
    n_hidden_channels,
    channel_mults=(1, 2, 2, 2),
    n_res_blocks=2,
    attention_resolutions=(16,),
    n_embedding_features=256,
    dropout_rate=0.0,
    n_heads=4,
    n_classes=None,
    *,
    rngs,
  ):
    """Construct a UNet.

    Args:
      image_size: size of the image, e.g., (32, 32, 3)
      n_hidden_channels: base number of hidden channels; each resolution
        level uses n_hidden_channels * channel_mults[level]
      channel_mults: per-level channel multiplier, one entry per
        resolution level (levels after the first are downsampled by 2x)
      n_res_blocks: number of ResBlocks per resolution level
      attention_resolutions: spatial resolutions (H == W) at which to
        insert an AttentionBlock after each ResBlock
      n_embedding_features: dimensionality of the time/class embedding
      dropout_rate: float
      n_heads: number of attention heads
      n_classes: number of classes to condition on, or None for an
        unconditional model. Same contract as DiT: __call__ requires
        context (integer class labels) iff n_classes is set.
      rngs: random keys
    """
    self.image_size = image_size
    self.n_in_channels = image_size[-1]
    self.n_embedding_features = n_embedding_features
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
    self.in_conv = nnx.Conv(
      self.n_in_channels, n_hidden_channels, (3, 3), padding="SAME", rngs=rngs
    )

    down_res_blocks = []
    down_attn_blocks = []
    downsamples = []
    channels = n_hidden_channels
    resolution = image_size[0]
    skip_channels = [channels]
    for level, mult in enumerate(channel_mults):
      out_channels = n_hidden_channels * mult
      level_res_blocks = []
      level_attn_blocks = []
      for _ in range(n_res_blocks):
        level_res_blocks.append(
          ResBlock(
            channels,
            out_channels,
            n_embedding_features,
            dropout_rate=dropout_rate,
            rngs=rngs,
          )
        )
        channels = out_channels
        if resolution in attention_resolutions:
          level_attn_blocks.append(AttentionBlock(channels, n_heads, rngs=rngs))
        else:
          level_attn_blocks.append(None)
        skip_channels.append(channels)
      down_res_blocks.append(tuple(level_res_blocks))
      down_attn_blocks.append(tuple(level_attn_blocks))
      if level < len(channel_mults) - 1:
        downsamples.append(Downsample(channels, rngs=rngs))
        resolution //= 2
        skip_channels.append(channels)
      else:
        downsamples.append(None)
    self.down_res_blocks = tuple(down_res_blocks)
    self.down_attn_blocks = tuple(down_attn_blocks)
    self.downsamples = tuple(downsamples)

    self.mid_res_block1 = ResBlock(
      channels,
      channels,
      n_embedding_features,
      dropout_rate=dropout_rate,
      rngs=rngs,
    )
    self.mid_attn = AttentionBlock(channels, n_heads, rngs=rngs)
    self.mid_res_block2 = ResBlock(
      channels,
      channels,
      n_embedding_features,
      dropout_rate=dropout_rate,
      rngs=rngs,
    )

    up_res_blocks = []
    up_attn_blocks = []
    upsamples = []
    for level, mult in reversed(list(enumerate(channel_mults))):
      out_channels = n_hidden_channels * mult
      level_res_blocks = []
      level_attn_blocks = []
      for _ in range(n_res_blocks + 1):
        skip_ch = skip_channels.pop()
        level_res_blocks.append(
          ResBlock(
            channels + skip_ch,
            out_channels,
            n_embedding_features,
            dropout_rate=dropout_rate,
            rngs=rngs,
          )
        )
        channels = out_channels
        if resolution in attention_resolutions:
          level_attn_blocks.append(AttentionBlock(channels, n_heads, rngs=rngs))
        else:
          level_attn_blocks.append(None)
      up_res_blocks.append(tuple(level_res_blocks))
      up_attn_blocks.append(tuple(level_attn_blocks))
      if level > 0:
        upsamples.append(Upsample(channels, rngs=rngs))
        resolution *= 2
      else:
        upsamples.append(None)
    self.up_res_blocks = tuple(up_res_blocks)
    self.up_attn_blocks = tuple(up_attn_blocks)
    self.upsamples = tuple(upsamples)

    self.out_norm = nnx.GroupNorm(channels, num_groups=8, rngs=rngs)
    self.out_conv = nnx.Conv(
      channels, self.n_in_channels, (3, 3), padding="SAME", rngs=rngs
    )

  def __call__(
    self, inputs: jax.Array, times: jax.Array, context: jax.Array = None
  ) -> jax.Array:
    """Transform inputs through the UNet.

    Args:
      inputs: input in image form, shape (batch, H, W, C)
      times: one-dimensional array, shape (batch,)
      context: integer class labels, shape (batch,), required if this
        UNet was constructed with n_classes set; must be None otherwise.

    Returns:
      returns a jax.Array, same shape as inputs

    Raises:
      ValueError: if context is None but n_classes was set, or if context
        is given but n_classes was not set.
    """
    if self.n_classes is not None and context is None:
      raise ValueError(
        "this UNet was constructed with n_classes set, so context "
        "(integer class labels) must be provided"
      )
    if self.n_classes is None and context is not None:
      raise ValueError(
        "this UNet was constructed without n_classes, so context must "
        "be None -- pass n_classes at construction to condition on it"
      )

    embedding = self.time_embedding(
      timestep_embedding(times, self.n_embedding_features)
    )
    if context is not None:
      embedding = embedding + self.class_embedding(context)

    hidden = self.in_conv(inputs)
    skips = [hidden]
    for level in range(len(self.down_res_blocks)):
      for res_block, attn_block in zip(
        self.down_res_blocks[level], self.down_attn_blocks[level]
      ):
        hidden = res_block(hidden, embedding)
        if attn_block is not None:
          hidden = attn_block(hidden)
        skips.append(hidden)
      if self.downsamples[level] is not None:
        hidden = self.downsamples[level](hidden)
        skips.append(hidden)

    hidden = self.mid_res_block1(hidden, embedding)
    hidden = self.mid_attn(hidden)
    hidden = self.mid_res_block2(hidden, embedding)

    for level in range(len(self.up_res_blocks)):
      for res_block, attn_block in zip(
        self.up_res_blocks[level], self.up_attn_blocks[level]
      ):
        skip = skips.pop()
        hidden = res_block(jnp.concatenate([hidden, skip], axis=-1), embedding)
        if attn_block is not None:
          hidden = attn_block(hidden)
      if self.upsamples[level] is not None:
        hidden = self.upsamples[level](hidden)

    hidden = jax.nn.silu(self.out_norm(hidden))
    outputs = self.out_conv(hidden)
    return outputs


def SmallUNet(image_size, **kwargs):
  return UNet(
    image_size,
    n_hidden_channels=64,
    channel_mults=(1, 2, 2),
    n_res_blocks=2,
    **kwargs,
  )


def BaseUNet(image_size, **kwargs):
  return UNet(
    image_size,
    n_hidden_channels=128,
    channel_mults=(1, 2, 2, 2),
    n_res_blocks=2,
    **kwargs,
  )


def LargeUNet(image_size, **kwargs):
  return UNet(
    image_size,
    n_hidden_channels=192,
    channel_mults=(1, 1, 2, 2, 4),
    n_res_blocks=3,
    **kwargs,
  )
