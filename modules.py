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
    self.fn = fn
    self.norm = nnx.LayerNorm(dim, rngs=rngs)

  def __call__(self, x: jax.Array, *args: Any, **kwds: Any) -> Any:
    norm = self.norm(x)
    return self.fn(x)

class Block(nnx.Module):
  """A basic convolutional block with projection, group normalization, and SiLU activation."""
  def __init__(self, in_features:int, out_features:int, rngs:nnx.Rngs, groups:int = 8):
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