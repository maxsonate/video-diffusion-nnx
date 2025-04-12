import math
from functools import partial
from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from video_diffusion_pytorch.text import BERT_MODEL_DIM, bert_embed, tokenize

# Import custom modules
from modules import (Block, EinopsToAndFrom, Identity, Linear,
                     MultiheadAttention, PreNorm, RelativePositionBias,
                     Residual, ResnetBlock, SinusoidalPosEmb,
                     SpatialLinearAttention)
# Import helpers from utils.py
from utils import Downsample, Upsample, default, exists, is_odd, prob_mask_like


class Unet3D(nnx.Module):
    """
    A 3D U-Net model adapted for video diffusion tasks, utilizing Flax NNX.

    This network takes noisy video frames and corresponding time steps (and optional conditioning)
    and outputs a predicted noise tensor of the same shape as the input video.
    It follows a typical U-Net architecture with downsampling and upsampling paths,
    incorporating ResNet blocks, spatial attention, and temporal attention.

    Args:
        dim (int): Base dimension (number of channels) for the initial convolution.
        rngs (nnx.Rngs): NNX PRNGKey stream for parameter initialization.
        channels (int, optional): Number of input video channels. Defaults to 3.
        dim_mults (Tuple[int, ...], optional): Multipliers for the dimension at each U-Net level.
                                             Defaults to (1, 2, 4, 8).
        num_frames (int, optional): Number of frames expected in the input tensor for relative
                                    positional bias calculation. Defaults to 16.
        frame_rate (int, optional): Frame rate (not directly used in architecture but potentially useful
                                    for conditioning later). Defaults to 30.
        num_classes (int | None, optional): Number of classes for class conditioning. If provided,
                                          `class_embed_dim` must also be set. Defaults to None.
        class_embed_dim (int | None, optional): Dimension for class embeddings. Defaults to None.
        cond_dim (int | None, optional): Dimension of the conditioning vector (e.g., text embeddings).
                                       If provided, conditioning is expected during the forward pass.
                                       Defaults to None.
        use_bert_text_cond (bool, optional): Flag indicating if BERT embeddings are used (influences
                                           `cond_dim`). Defaults to False.
        cond_on_text (bool, optional): Whether to condition on text embeddings. Defaults to True.
        attn_heads (int, optional): Number of heads for MultiheadAttention. Defaults to 8.
        attn_dim_head (int, optional): Dimension of each attention head. Defaults to 64.
        use_sparse_linear_attn (bool, optional): Whether to use SpatialLinearAttention instead of
                                               standard MultiheadAttention for spatial attention.
                                               Defaults to True.
        resnet_groups (int, optional): Number of groups for Group Normalization in ResNet blocks.
                                     Defaults to 8.
    """

    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        dim_mults=(1, 2, 4, 8),  # Channel multipliers
        cond_dim=None,  # Conditioning embedding dimension
        out_dim=None,  # Out dimension, default to 3 (channels)
        channels=3,  # Images channels
        attn_heads=8,  # Attention Resolution
        attn_dim_head=32,  # Attention head dimension
        use_bert_text_cond=False,  # Use bert tokens to condition on text
        init_dim=None,  # Default to base channel dim
        init_kernel_size=7,  # initial kernel size
        use_sparse_linear_attn=True,  # Use sparse linear attention
        block_type="resnet",  # Block type, default to Resnet
        resnet_groups=8,  # Resnet groups
        log_dims=False,
    ):

        # TBD: What is sparse linear attention?
        self.channels = channels
        self.log_dims = log_dims
        # TBD: figure out how to get rotary embedding here.
        rotary_emb = None
        # TBD: double check if the attn_dim_head is what is expected:
        # Temporal attention and its relative positional encoding:
        # TBD: One issue is that in nnx, the conv is over the last dim, but in pytorch, it's over the 2nd dim
        # temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c',
        temporal_attn = lambda dim: EinopsToAndFrom(
            "b f h w c",
            "b (h w) f c",
            MultiheadAttention(
                in_features=dim,
                dim=attn_dim_head,
                num_heads=attn_heads,
                rotary_emb=rotary_emb,
                rngs=rngs,
            ),
        )

        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32, rngs=rngs
        )
        # Realistically will not be able to generate that many frames of video yet!

        init_dim = default(init_dim, dim)
        assert init_dim is not None
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2

        # Initial convolution:
        # I assume the input is BxHeightxWeightxChannels. Double check this.
        self.init_conv = nnx.Conv(
            channels,
            init_dim,
            kernel_size=(1, init_kernel_size, init_kernel_size),
            rngs=rngs,
        )

        # initial temporal attention:
        self.init_temporal_attn = Residual(
            PreNorm(init_dim, temporal_attn(init_dim), rngs=rngs)
        )

        # Dimensions
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time Conditioning. TBD: How is this different from the temporal attn?
        time_dim = dim * 4  # This is a common choice in diffusion models.
        self.time_mlp = nnx.Sequential(
            SinusoidalPosEmb(dim),
            nnx.Linear(dim, time_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(time_dim, time_dim, rngs=rngs),
        )

        # Text conditioning
        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim
        self.null_cond_emb = (
            nnx.Param(
                jnp.astype(
                    jax.random.randint(
                        jax.random.PRNGKey(0), (1, cond_dim), minval=1, maxval=cond_dim
                    ),
                    jnp.float32,
                )
            )
            if self.has_cond
            else 0.0
        )  # TBD:Why this parameter is needed?
        cond_dim = time_dim + int(cond_dim or 0)

        # Layers
        num_resolutions = len(in_out)

        # Block type
        block_klass = partial(ResnetBlock, groups=resnet_groups, rngs=rngs)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        self.downs = []
        self.ups = []
        # Modules for all layers
        # TBD: Double check if D = 32 is a proper value.
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                [
                    block_klass_cond(dim_in, dim_out),  # resnet
                    block_klass_cond(dim_out, dim_out),  # resnet
                    (
                        Residual(
                            PreNorm(
                                dim_out,
                                SpatialLinearAttention(
                                    dim_out, heads=attn_heads, D=32, rngs=rngs
                                ),
                                rngs=rngs,
                            )
                        )
                        if use_sparse_linear_attn
                        else Identity()
                    ),  # Spatial Attention
                    Residual(
                        PreNorm(dim_out, temporal_attn(dim_out), rngs=rngs)
                    ),  # Temportal attention
                    (
                        Downsample(dim_out, rngs=rngs) if not is_last else Identity()
                    ),  # Identity
                ]
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)  # Resnet

        # TBD: nnx conv channels is last dim, but in pytorch, it's 2nd dim.
        # spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', MultiheadAttention(in_features=mid_dim,
        spatial_attn = EinopsToAndFrom(
            "b f h w c",
            "b f (h w) c",
            MultiheadAttention(
                in_features=mid_dim, dim=attn_dim_head, num_heads=attn_heads, rngs=rngs
            ),
        )
        self.mid_spatial_attn = Residual(
            PreNorm(mid_dim, spatial_attn, rngs=rngs)
        )  # Spatial Attention
        self.mid_temporal_attn = Residual(
            PreNorm(mid_dim, temporal_attn(mid_dim), rngs=rngs)
        )  # Temporal Attention

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)  # Resent

        # Upsampling:
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                [
                    block_klass_cond(dim_out * 2, dim_in),  # Resnet
                    block_klass_cond(dim_in, dim_in),  # Resnet
                    (
                        Residual(
                            PreNorm(
                                dim_in,
                                SpatialLinearAttention(
                                    dim_in, heads=attn_heads, D=32, rngs=rngs
                                ),
                                rngs=rngs,
                            )
                        )
                        if use_sparse_linear_attn
                        else Identity()
                    ),  # Spatial Attention
                    Residual(
                        PreNorm(dim_in, temporal_attn(dim_in), rngs=rngs)
                    ),  # Temporal attention
                    (
                        Upsample(dim_in, rngs=rngs) if not is_last else Identity()
                    ),  # Upsample
                ]
            )

        out_dim = default(out_dim, channels)

        self.final_conv = nnx.Sequential(
            block_klass(dim * 2, dim), nnx.Conv(dim, out_dim, 1, rngs=rngs)  # Resnet
        )

    def forward_with_cond_scale(self, *args, cond_scale=2.0, **kwargs):
        logits = self.__call__(*args, null_cond_prob=0.0, **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.__call__(*args, null_cond_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def __call__(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.0,
        focus_present_mask=None,
        prob_focus_present=0.0,  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (
            self.has_cond and not exists(cond)
        ), "cond must be passed in if cond_dim specified"

        focus_present_mask = default(
            focus_present_mask,
            lambda: prob_mask_like((x.shape[0],), prob_focus_present),
        )
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2])
        x = rearrange(x, "b c f h w -> b f h w c")
        # Apply initial convolution
        x = self.init_conv(x)
        # Apply initial temporal attention
        x = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)

        r = x  # clone is not needed bc in jax numpy all arrays are immutable.

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # Classifier free guidance:
        if self.has_cond:
            # TODO: The conversion from torch to jnp needs to be handled properly outside of the module.
            # cond = jnp.array(cond.detach().cpu().numpy())
            mask = prob_mask_like((x.shape[0],), null_cond_prob)
            cond = jnp.where(
                rearrange(mask, "b -> b 1"), self.null_cond_emb, cond
            )  # TBD: I don't understand this properly, double check.
            t = jnp.concat((t, cond), axis=-1)

        h = []

        # Iterate over the downs blocks:
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)  # Resnet
            x = block2(x, t)  # Resnet
            x = spatial_attn(x)  # spatial attention
            x = temporal_attn(
                x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask
            )  # Temporal Attention
            h.append(x)
            if self.log_dims:
                print(f"pre downsample:{x.shape}")
            x = downsample(x)  # Downsample

        # Mid blocks:
        x = self.mid_block1(x, t)  # Resnet
        x = self.mid_spatial_attn(x)  # Spatial Attention
        x = self.mid_temporal_attn(
            x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask
        )  # Temporal Attention
        x = self.mid_block2(x, t)  # Resnet

        # Iterate over the ups blocks
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            if self.log_dims:
                print(f"ups pre concat:{x.shape}")
            x = jnp.concat(
                (x, h.pop()), axis=-1
            )  # The corresponding feature map from the contracting path is concatenated with the current feature map
            if self.log_dims:
                print(f"ups: block1 :{x.shape}")
            x = block1(x, t)  # Resnet
            x = block2(x, t)  # Resent
            x = spatial_attn(x)  # Spatial Attention

            x = temporal_attn(
                x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask
            )  # Temporal attention
            if self.log_dims:
                print(f"pre upsample:{x.shape}")
            x = upsample(x)  # Upsample

        x = jnp.concat((x, r), axis=-1)
        if self.log_dims:
            print(f"final conv:{x.shape}")
        return self.final_conv(x)
