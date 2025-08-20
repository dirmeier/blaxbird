# from collections.abc import Sequence
#
# import chex
# import jax
# import jax.numpy as jnp
# from einops import rearrange
# from flax import nnx
# from flax.nnx import dot_product_attention, rnglib
#
#
# class _DotProductAttention(nnx.Module):
#   def __init__(self, n_heads):
#     self.n_heads = n_heads
#
#   def __call__(self, inputs):
#     B, H, W, C = inputs.shape
#     chex.assert_equal(C % (3 * self.n_heads), 0)
#     q, k, v = jnp.split(inputs, 3, axis=3)
#     outputs = dot_product_attention(
#       rearrange(q, "b h w (c heads) -> b (h w) heads c", heads=self.n_heads),
#       rearrange(k, "b h w (c heads) -> b (h w) heads c", heads=self.n_heads),
#       rearrange(v, "b h w (c heads) -> b (h w) heads c", heads=self.n_heads),
#     )
#     outputs = rearrange(
#       outputs,
#       "b (h w) heads c -> b h w (heads c)",
#       heads=self.n_heads,
#       h=H,
#       w=W,
#     )
#     return outputs
#
#
# class _AttentionBlock(nnx.Module):
#   n_heads: int
#   n_groups: int
#
#   def __init__(self, n_features, n_heads, n_groups, rngs: rnglib.Rngs):
#     self.n_heads = n_heads
#     self.n_groups = n_groups
#     self.group_norm = nnx.GroupNorm(n_features, n_groups, rngs=rngs)
#     self.conv = nnx.Conv(
#       n_features,
#       n_features * 3,
#       kernel_size=(1, 1),
#       strides=(1, 1),
#       padding="SAME",
#       rngs=rngs,
#     )
#     self.out_conv = nnx.Conv(
#       n_features,
#       n_features,
#       kernel_size=(1, 1),
#       strides=(1, 1),
#       padding="SAME",
#       kernel_init=nnx.initializers.zeros,
#       rngs=rngs,
#     )
#
#   def __call__(self, inputs, is_training):
#     hidden = inputs
#     hidden = self.group_norm(hidden)
#     # input projection (replacing the MLP on conventional attention)
#     hidden = self.conv(hidden)
#     # attention, we don't push through linear layers since we have the
#     # convolution above outputting 3 times the layers which we use as
#     # k, q, v
#     hidden = _DotProductAttention(self.n_heads)(hidden)
#     # output projection (replacing the MLP on conventional attention)
#     outputs = self.out_conv(hidden)
#     return outputs + inputs
#
#
# class _Downsample(nnx.Module):
#   def __init__(
#     self, n_features, use_conv, kernel_size, stride, rngs: rnglib.Rngs
#   ):
#     if use_conv:
#       self.fn = nnx.Conv(
#         n_features,
#         n_features,
#         kernel_size=(kernel_size, kernel_size),
#         strides=(stride, stride),
#         padding=kernel_size // 2,
#         rngs=rngs,
#       )
#     else:
#       self.fn = lambda x: nnx.avg_pool(
#         x,
#         window_shape=(stride, stride),
#         strides=(stride, stride),
#       )
#
#   def __call__(self, inputs):
#     return self.fn(inputs)
#
#
# class _Upsample(nnx.Module):
#   def __init__(
#     self, n_features, use_conv, kernel_size, stride, rngs: rnglib.Rngs
#   ):
#     self.use_conv = use_conv
#     if self.use_conv:
#       self.conv = nnx.Conv(
#         n_features,
#         n_features,
#         kernel_size=(kernel_size, kernel_size),
#         strides=(stride, stride),
#         padding="SAME",
#         rngs=rngs,
#       )
#
#   def __call__(self, inputs):
#     B, H, W, C = inputs.shape
#     outputs = jax.image.resize(
#       inputs,
#       (B, H * 2, W * 2, C),
#       method="nearest",
#     )
#     if self.use_conv:
#       outputs = self.conv(outputs)
#     return outputs
#
#
# class _ConditionalResidualBlock(nnx.Module):
#   n_out_channels: int
#   dropout_rate: float
#   kernel_size: int
#   n_groups: int
#
#   def __init__(self, n_channels, n_groups, *, dropout_rate, kernel_size, rngs):
#     self.group_norm1 = nnx.GroupNorm(n_channels, n_groups, rngs=rngs)
#     self.conv1 = nnx.Conv(
#       n_channels,
#       n_channels,
#       kernel_size=(kernel_size, kernel_size),
#       strides=(1, 1),
#       padding="SAME",
#       rngs=rngs,
#     )
#     self.embedding = nnx.Linear()
#
#   @nn.compact
#   def __call__(self, inputs, sigma, is_training):
#     hidden = inputs
#     # convolution with pre-layer norm
#
#     hidden = self.conv1(nnx.silu(self.group_norm1(hidden)))
#
#     embedding = nn.Dense(self.n_out_channels)(sigma)
#     hidden += embedding[:, None, None, :]
#
#     # convolution with pre-layer norm and dropout
#     hidden = nn.GroupNorm(num_groups=self.n_groups)(hidden)
#     hidden = nn.silu(hidden)
#     hidden = nn.Dropout(self.dropout_rate)(
#       hidden, deterministic=not is_training
#     )
#     hidden = nn.Conv(
#       self.n_out_channels,
#       kernel_size=(self.kernel_size, self.kernel_size),
#       strides=(1, 1),
#       padding="SAME",
#       kernel_init=nn.initializers.zeros,
#     )(hidden)
#
#     if inputs.shape[-1] != self.n_out_channels:
#       residual = nn.Conv(
#         self.n_out_channels,
#         kernel_size=(1, 1),
#         strides=(1, 1),
#         padding="SAME",
#       )(inputs)
#     else:
#       residual = inputs
#
#     return hidden + residual
#
#
# class UNet(nn.Module):
#   n_channels: int
#   n_out_channels: int
#   channel_multipliers: Sequence[int]
#   n_resnet_blocks: int
#   n_classes: int | None = None
#   n_embedding: int = 256
#   attention_resolutions: Sequence[int] = ()
#   n_attention_heads: int = 2
#   kernel_size: int = 3
#   dropout_rate: float = 0.1
#   use_conv_in_resize: bool = True
#   n_groups: int = 32
#
#   @nn.compact
#   def __call__(
#     self,
#     inputs,
#     times,
#     context=None,
#     is_training=False,
#     **kwargs,
#   ):
#     # the input is assumed to be channel last (as is the convention in Flax)
#     # B, H, W, C = inputs.shape
#     hidden = inputs
#     # embed the time points and the conditioning variables
#     times = nn.Sequential(
#       [
#         lambda x: timestep_embedding(times, self.n_embedding),
#         nn.Dense(self.n_embedding),
#         nn.silu,
#         nn.Dense(self.n_embedding),
#       ]
#     )(times)
#     if context is not None and self.n_classes is not None:
#       context = nn.Embed(self.n_classes + 1, self.n_embedding)(context)
#       times = times + context
#     times = nn.silu(times)
#     # lift data
#     hidden = nn.Conv(
#       self.n_channels,
#       kernel_size=(self.kernel_size, self.kernel_size),
#       strides=(1, 1),
#       padding="SAME",
#     )(hidden)
#
#     hs = [hidden]
#     # downsampling UNet blocks
#     for level, channel_mult in enumerate(self.channel_multipliers):
#       n_outchannels = channel_mult * self.n_channels
#       for _ in range(self.n_resnet_blocks):
#         hidden = _ConditionalResidualBlock(
#           n_out_channels=n_outchannels,
#           dropout_rate=self.dropout_rate,
#           kernel_size=self.kernel_size,
#           n_groups=self.n_groups,
#         )(hidden, times, is_training)
#         if hidden.shape[1] in self.attention_resolutions:
#           hidden = _AttentionBlock(
#             n_heads=self.n_attention_heads, n_groups=self.n_groups
#           )(hidden, is_training)
#         hs.append(hidden)
#       if level != len(self.channel_multipliers) - 1:
#         hidden = _Downsample(
#           use_conv=self.use_conv_in_resize,
#           kernel_size=self.kernel_size,
#         )(hidden, is_training)
#         hs.append(hidden)
#
#     # middle UNet block
#     n_outchannels = self.channel_multipliers[-1] * self.n_channels
#     for i in range(2):
#       hidden = _ConditionalResidualBlock(
#         n_out_channels=n_outchannels,
#         dropout_rate=self.dropout_rate,
#         kernel_size=self.kernel_size,
#         n_groups=self.n_groups,
#       )(hidden, times, is_training)
#       if i < self.n_resnet_blocks:
#         hidden = _AttentionBlock(
#           n_heads=self.n_attention_heads, n_groups=self.n_groups
#         )(hidden, is_training)
#
#     # upsampling UNet block
#     for level, channel_mult in reversed(
#       list(enumerate(self.channel_multipliers))
#     ):
#       n_outchannels = channel_mult * self.n_channels
#       for idx in range(self.n_resnet_blocks + 1):
#         hidden = jnp.concatenate([hidden, hs.pop()], axis=-1)
#         hidden = _ConditionalResidualBlock(
#           n_out_channels=n_outchannels,
#           dropout_rate=self.dropout_rate,
#           kernel_size=self.kernel_size,
#           n_groups=self.n_groups,
#         )(hidden, times, is_training)
#         if hidden.shape[1] in self.attention_resolutions:
#           hidden = _AttentionBlock(
#             n_heads=self.n_attention_heads, n_groups=self.n_groups
#           )(hidden, is_training)
#         if level and idx == self.n_resnet_blocks:
#           hidden = _Upsample(
#             use_conv=self.use_conv_in_resize,
#             kernel_size=self.kernel_size,
#           )(hidden, is_training)
#
#     outputs = nn.Sequential(
#       [
#         nn.GroupNorm(self.n_groups),
#         nn.silu,
#         nn.Conv(
#           self.n_out_channels,
#           kernel_size=(
#             self.kernel_size,
#             self.kernel_size,
#           ),
#           strides=(1, 1),
#           padding="SAME",
#           kernel_init=nn.initializers.zeros,
#         ),
#       ]
#     )(hidden)
#     chex.assert_equal_size([inputs, outputs])
#     chex.assert_equal(len(hs), 0)
#     return outputs
