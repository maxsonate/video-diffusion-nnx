from flax import nnx
import jax
import jax.numpy as jnp
import math
from einops import rearrange
from typing import Any


class Linear(nnx.Module):
  """A standard linear layer."""
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b


class Residual(nnx.Module):
  """Adds the input to the output of a given function (residual connection)."""
  def __init__(self, fn:nnx.Module):
    self.fn = fn

  def __call__(self, x:jax.Array, *args: Any, **kwds: Any) -> Any:
    return self.fn(x, *args, **kwds) + x
  

class SinusoidalPosEmb(nnx.Module):
  """Generates sinusoidal positional embeddings."""
  def __init__(self, dim):
    self.dim = dim

  def __call__(self, x:jax.Array):

    half_dim = self.dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)

    emb = x[..., None] * emb[None, :]

    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

    return emb

class EinopsToAndFrom(nnx.Module):
  """Wraps a function, reshaping the input before and after applying it using einops."""
  def __init__(self, from_einops, to_einops, fn):
    self.from_einops = from_einops
    self.to_einops = to_einops
    self.fn = fn

  def __call__(self, x, **kwargs):
    shape = x.shape
    reconstitue_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
    x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
    x = self.fn(x, **kwargs)
    x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitue_kwargs)
    return x



class SpatialLinearAttention(nnx.Module):
  """Implements spatial linear attention mechanism."""
  def __init__(self, dim:int, heads, D:int, rngs:nnx.Rngs):
    self.scale = D ** -0.5
    self.heads = heads
    hD = D * heads

    self.q = nnx.Conv(dim,
                      hD,
                      kernel_size = 1,
                      use_bias=False,
                      rngs = rngs)
    self.k = nnx.Conv(dim,
                      hD,
                      kernel_size = 1,
                      use_bias=False,
                      rngs = rngs)
    self.v = nnx.Conv(dim,
                      hD,
                      kernel_size = 1,
                      use_bias=False,
                      rngs = rngs)

    self.to_out = nnx.Conv(hD,
                           dim,
                           kernel_size = 1,
                           use_bias=False,
                           rngs = rngs)
# b, c, f, h, w = x.shape
# batch, channel, frame, height, width
  def __call__(self, x:jax.Array)->jax.Array:
    #TBD nnx conv has channels as the last dim, but pytorch has it as the 2nd:
    # b, c, f, h, w = x.shape
    b, f, h, w, c = x.shape
    # print(f'Spatial Attention 1: {x.shape}')
    #TBD nnx conv has channels as the last dim, but pytorch has it as the 2nd:
    # x = rearrange(x, 'b c f h w -> (b f) h w c')
    x = rearrange(x, 'b f h w c -> (b f) h w c')

    # h: num heads, B : batch size, F : frame size, H: height, W: width, D: channels
    # print(f'Spatial Attention 2: {x.shape}')
    q_BFxHxWxhD = self.q(x)
    q_BFxhxDxHW = rearrange(q_BFxHxWxhD, 'b x y (h c) -> b h c (x y)', h = self.heads)
    q_BFxhxDxHW = nnx.softmax(q_BFxhxDxHW, axis=-2)
    q = q_BFxhxDxHW * self.scale

    k_BFxHxWxhD = self.k(x)
    k_BFxhxDxHW = rearrange(k_BFxHxWxhD, 'b x y (h c) -> b h c (x y)', h = self.heads)
    k_BFxhxDxHW = nnx.softmax(k_BFxhxDxHW, axis= -1)

    v_BFxHxWxhD = self.v(x)
    v_BFxhxDxHW = rearrange(v_BFxHxWxhD, 'b x y (h c) -> b h c (x y)', h = self.heads)

    context_BFxhxDxD = jnp.einsum('bhdn,bhen -> bhde', k_BFxhxDxHW, v_BFxhxDxHW)
    out_BFxhxDxHW = jnp.einsum('bhde,bhdn->bhen', context_BFxhxDxD, q_BFxhxDxHW)
    out_BFxHxWxhD = rearrange(out_BFxhxDxHW, 'b h c (x y) -> b x y (h c)',
                              h = self.heads,
                              x = h,
                              y = w
                              )

    out_BFxHxWxC = self.to_out(out_BFxHxWxhD)
    out_BxFxCxHxW = rearrange(out_BFxHxWxC, '(b f) h w c -> b f h w c', b = b)
    # print(f'Spatial Attention 4: {out_BxFxCxHxW.shape}')

    return out_BxFxCxHxW


class PreNorm(nnx.Module):
  """Applies layer normalization before applying a function."""
  def __init__(self, dim:int, fn:nnx.Module, rngs:nnx.Rngs):
    """
    Initializes the PreNorm module.

    Args:
        dim: The feature dimension for Layer Normalization.
        fn: The function/module to apply after normalization.
        rngs: NNX PRNGKey stream.
    """
    self.fn = fn
    self.norm = nnx.LayerNorm(dim, rngs=rngs)

  def __call__(self, x: jax.Array, *args: Any, **kwds: Any) -> Any:
    norm = self.norm(x)
    return self.fn(x)

class Block(nnx.Module):
  """A basic convolutional block with projection, group normalization, and SiLU activation."""
  def __init__(self, in_features:int, out_features:int, rngs:nnx.Rngs, groups:int = 8):
    """
    Initializes the Block module.

    Args:
        in_features: Number of input features/channels.
        out_features: Number of output features/channels.
        rngs: NNX PRNGKey stream.
        groups: Number of groups for Group Normalization. Defaults to 8.
    """
    self.proj = nnx.Conv(in_features,
                         out_features,
                          (1, 3, 3),
                         rngs=rngs) #TODO: Ensure that the kernel size is correct
    # Group normalization after the convolution
    self.norm = nnx.GroupNorm(out_features, num_groups=groups, rngs=rngs)

    self.act = nnx.silu

  def __call__(self, x:jax.Array, scale_shift = None, *args: Any, **kwds: Any) -> Any:
    x = self.proj(x)
    x = self.norm(x)

    if scale_shift is not None:
      scale, shift = scale_shift
      x = x * (scale + 1) + shift

    return self.act(x)


class ResnetBlock(nnx.Module):
  """A ResNet block combining convolutional blocks with optional time embeddings."""
  def __init__(self,
               in_features: int,
               out_features: int,
               rngs:nnx.Rngs,
               *,
               time_emb_dim = None,
               groups = 8):
    """
    Initializes the ResnetBlock module.

    Args:
        in_features: Number of input features/channels.
        out_features: Number of output features/channels.
        rngs: NNX PRNGKey stream.
        time_emb_dim: Dimension of the time embedding. If None, time embedding is not used. Defaults to None.
        groups: Number of groups for Group Normalization in the Blocks. Defaults to 8.
    """

    self.mlp = nnx.Sequential(nnx.silu,
                              nnx.Linear(
                                  in_features=time_emb_dim,
                                  out_features=out_features * 2,
                                  rngs=rngs)
                              ) if time_emb_dim is not None else None
    self.norm_1 = nnx.LayerNorm(out_features * 2, rngs=rngs)
    self.block_1 = Block(in_features=in_features,
                         out_features=out_features,
                         groups = groups,
                         rngs=rngs
                         )
    self.block_2 = Block(in_features=out_features,
                         out_features=out_features,
                         groups = groups,
                         rngs=rngs
                         )
    self.res_conv = nnx.Conv(in_features=in_features,
                             out_features=out_features,
                             kernel_size = 1,
                             rngs=rngs) if in_features != out_features else Identity()
    self.norm_2 = nnx.LayerNorm(out_features, rngs=rngs)


  def __call__(self, x:jax.Array, time_embed : jax.Array | None = None):

    scale_shift = None

    if self.mlp is not None:
      assert time_embed is not None, 'time emb must be passed in'

      time_embed = self.mlp(time_embed)
      time_embed = self.norm_1(time_embed)
      time_embed = rearrange(time_embed, 'b c -> b 1 1 1 c' ) # It is BxC, because we have one time stamp for each frame.
      # TODO: double check if the dimesion is correct here. channel is assumed to be dim 1.

      scale_shift = jnp.split(time_embed, 2, axis=-1)

    h = self.block_1(x, scale_shift = scale_shift)
    h = self.block_2(h)
    h = h + self.norm_2(self.res_conv(x))
    return h


# Multihead attention for temporal attention
class MultiheadAttention(nnx.Module):
  """Implements multi-head attention, potentially with rotary embeddings and relative position bias."""
  def __init__(self, in_features:int, dim:int, num_heads:int, rngs:nnx.Rngs, rotary_emb: Any | None = None):
    """
    Initializes the MultiheadAttention module.

    Args:
        in_features: Total dimension of the input features.
        dim: Dimension of each attention head's query, key, and value.
        num_heads: Number of attention heads.
        rngs: NNX PRNGKey stream.
        rotary_emb: Optional rotary embedding module. Defaults to None.
    """

    self.q = nnx.LinearGeneral(in_features=in_features,
                               out_features=(num_heads, dim),
                               rngs = rngs)

    self.k = nnx.LinearGeneral(in_features=in_features,
                               out_features=(num_heads, dim),
                               rngs = rngs)

    self.v = nnx.LinearGeneral(in_features=in_features,
                               out_features=(num_heads, dim),
                               rngs = rngs)

    self.out = nnx.LinearGeneral(in_features=(num_heads, dim),
                                 out_features=in_features,
                                 axis=(-2, -1),
                                 rngs=rngs)
    self.dim = dim
    self.rotary_emb = rotary_emb

  def __call__(self, x:jax.Array,
               focus_present_mask: jax.Array | None = None,
               pos_bias: jax.Array | None = None) -> jax.Array:
    # x is assumed to be ...fxd, where f is the time/frame dim.
    # K here indicates the number of attn heads to not be confused with Height.
    q_xxxFxKxD = self.q(x)
    k_xxxFxKxD = self.k(x)
    v_xxxFxKxD = self.v(x)
    F = x.shape[-2]

    # If all the batch samples are focuing on present:
    if focus_present_mask is not None and jnp.all(focus_present_mask):
      return self.out(v_xxxFxKxD)

    q_xxxFxKxD = q_xxxFxKxD / self.dim ** 0.5

    # TODO: Rotate positions into queries and keys for time attention
    # TODO: There is no jax implementaiton for this yet. Think about implementing it.
    if self.rotary_emb:
      q_xxxFxKxD = self.rotary_emb.rotate_queries_or_keys(q_xxxFxKxD)
      k_xxxFxKxD = self.rotary_emb.rotate_queries_or_keys(k_xxxFxKxD)


    qk_xxxKxFxF= jnp.einsum('...ihd,...jhd->...hij', q_xxxFxKxD, k_xxxFxKxD)
    attn_xxxKxFxF = jax.nn.softmax(qk_xxxKxFxF, axis=-1)

    # Add temporal attention mask:
    if focus_present_mask is not None and jnp.any(focus_present_mask):
      attend_all_mask = jnp.ones((F, F), dtype=jnp.int32)
      attend_self_mask = jnp.eye(F, dtype=jnp.int32)

      mask = jnp.where(rearrange(focus_present_mask, 'b -> b 1 1 1 1 1'),
                       rearrange(attend_self_mask, 'i j -> 1 1 1 1 i j'),
                       rearrange(attend_all_mask, 'i j -> 1 1 1 1 i j'))

      _NEG_INF = jnp.finfo(jnp.float32).min
      attn_xxxKxFxF = jnp.where(mask, attn_xxxKxFxF, _NEG_INF)

    # TODO: Double check that the below is correct.
    # Relative positional bias
    if pos_bias is not None:
      attn_xxxKxFxF += pos_bias

    attn_xxxFxKxD = jnp.einsum('...hij,...jhd->...ihd', attn_xxxKxFxF, v_xxxFxKxD)
    out = self.out(attn_xxxFxKxD)

    return out

# Relative Position Bias:
# This class computes the relative position bias for temporal attention mechanism
class RelativePositionBias(nnx.Module):
  """Computes relative positional bias for attention mechanisms."""
  def __init__(self,
             rngs:nnx.Rngs,
             heads:int = 8,
             num_buckets:int = 32,  #  Number of buckets to cluster relative position distances
             max_distance:int = 128): # Maximum distance to be considered for bucketing. Distances beyond this will be mapped to the last bucket
    """
    Initializes the RelativePositionBias module.

    Args:
        rngs: NNX PRNGKey stream.
        heads: Number of attention heads. Defaults to 8.
        num_buckets: Number of buckets to cluster relative position distances. Defaults to 32.
        max_distance: Maximum distance considered for bucketing. Distances beyond this map to the last bucket. Defaults to 128.
    """
    self.num_buckets = num_buckets
    self.max_distance = max_distance
    self.relative_attention_bias = nnx.Embed(num_buckets, heads, rngs=rngs)

  @staticmethod
  def _relative_position_bucket(relative_position,
                                num_buckets = 32,
                                max_distance = 128
                                ):

    # i) Divides the buckets into two : first have for pos and 2nd for neg
    # ii) for the half of the buckets, assigment is identity.
    # iii) for the remaining, it is a log schedule.
    # iv) caps to max bucket.
    ret = 0
    n = -relative_position

    num_buckets //= 2
    ret += jnp.astype((n < 0), jnp.int32) * num_buckets
    n = jnp.abs(n)

    max_exact = num_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + jnp.astype(jnp.log(
        jnp.astype(n, jnp.float32) / max_exact) / math.log(
            max_distance / max_exact) * (num_buckets - max_exact), jnp.int32)

    val_if_large = jnp.minimum(val_if_large,
                               jnp.full_like(val_if_large, num_buckets - 1))

    ret += jnp.where(is_small, n, val_if_large)
    return ret

  def __call__(self, n):

    q_pos = jnp.arange(n, dtype=jnp.int32)
    k_pos = jnp.arange(n, dtype=jnp.int32)

    rel_pos = rearrange(q_pos,'i -> i 1') - rearrange(k_pos, 'j -> 1 j')
    rp_buckets = self._relative_position_bucket(rel_pos)

    emb = self.relative_attention_bias(rp_buckets)

    return rearrange(emb, 'i j h -> h i j')


class Identity(nnx.Module):
  """An identity layer that returns its input unchanged."""
  def __call__(self, x: jax.Array, *args: Any, **kwds: Any) -> jax.Array:
    return x

