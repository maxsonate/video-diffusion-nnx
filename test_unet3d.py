import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import unet3d   # Import the Unet3D module

@pytest.fixture
def rngs():
  """Provides nnx.Rngs for tests."""
  return nnx.Rngs(0)

def test_unet3d(rngs):
    """Tests the Unet3D module."""
    dim = 16 # Base dimension
    channels = 3
    dim_mults = (1, 2, 4, 8) # Dimension multipliers for down/up blocks
    cond_dim = 32 # Example conditioning dimension
    batch_size = 1
    frames = 4
    height = 16 # Use powers of 2 for easier down/upsampling
    width = 16

    input_shape = (batch_size, channels, frames, height, width)
    output_shape = (batch_size, frames, height, width, channels)
    key = jax.random.PRNGKey(12)
    x = jax.random.normal(key, input_shape)

    time_key, cond_key = jax.random.split(key)
    # Example time steps (batch_size,)
    time = jax.random.randint(time_key, (batch_size,), 0, 1000)
    # Example conditioning vectors (batch_size, cond_dim)
    cond = jax.random.normal(cond_key, (batch_size, cond_dim))

    # Initialize Unet3D module
    unet_module = unet3d.Unet3D(
        dim=dim,
        channels=channels,
        dim_mults=dim_mults,
        cond_dim=cond_dim,
        rngs=rngs
    )

    # Run the forward pass
    # Assuming unet_module takes JAX arrays directly
    output = unet_module(x, time, cond=cond)

    # Assertions
    assert output.shape == output_shape, f"Expected shape {output_shape}, but got {output.shape}"
    assert output.dtype == x.dtype

    # Test without conditioning
    unet_module_no_cond = unet3d.Unet3D(
        dim=dim,
        channels=channels,
        dim_mults=dim_mults,
        cond_dim=None, # Set cond_dim to None
        rngs=rngs
    )
    output_no_cond = unet_module_no_cond(x, time, cond=None)
    assert output_no_cond.shape == output_shape
    assert output_no_cond.dtype == x.dtype 