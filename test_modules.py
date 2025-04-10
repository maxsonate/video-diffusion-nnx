import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import modules  # Import the module containing your classes

@pytest.fixture
def rngs():
  """Provides nnx.Rngs for tests."""
  return nnx.Rngs(0)

def test_linear(rngs):
  """Tests the Linear module."""
  din, dout = 10, 5
  batch_size = 4
  input_shape = (batch_size, din)
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(key, input_shape)

  # Initialize the Linear module
  linear_module = modules.Linear(din=din, dout=dout, rngs=rngs)

  # Run the forward pass
  output = linear_module(x)

  # Assertions
  assert output.shape == (batch_size, dout)
  assert output.dtype == x.dtype 

def test_residual(rngs):
    """Tests the Residual module."""
    dim = 10
    batch_size = 4
    input_shape = (batch_size, dim)
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, input_shape)

    # Define a simple function (e.g., Linear layer)
    mock_fn = modules.Linear(din=dim, dout=dim, rngs=rngs)

    # Initialize Residual module
    residual_module = modules.Residual(fn=mock_fn)

    # Run the forward pass
    output = residual_module(x)
    expected_output = mock_fn(x) + x

    # Assertions
    assert output.shape == input_shape
    assert output.dtype == x.dtype
    assert jnp.allclose(output, expected_output)

def test_sinusoidal_pos_emb():
    """Tests the SinusoidalPosEmb module."""
    dim = 32
    batch_size = 4
    key = jax.random.PRNGKey(2)
    x = jax.random.uniform(key, (batch_size,)) * 100 # Example time steps

    # Initialize SinusoidalPosEmb module
    pos_emb_module = modules.SinusoidalPosEmb(dim=dim)

    # Run the forward pass
    output = pos_emb_module(x)

    # Assertions
    assert output.shape == (batch_size, dim)
    assert output.dtype == x.dtype

def test_einops_to_and_from(rngs):
    """Tests the EinopsToAndFrom module."""
    b, c, h, w = 2, 3, 4, 5
    input_shape = (b, c, h, w)
    key = jax.random.PRNGKey(3)
    x = jax.random.normal(key, input_shape)

    # Example function (identity in this case, could be a layer)
    def identity_fn(arr):
        return arr

    # Define einops patterns
    from_pattern = 'b c h w'
    to_pattern = '(b h w) c' # Example: flatten spatial dims into batch

    # Initialize EinopsToAndFrom module
    einops_module = modules.EinopsToAndFrom(from_einops=from_pattern,
                                            to_einops=to_pattern,
                                            fn=identity_fn)

    # Run the forward pass
    output = einops_module(x)

    # Assertions
    assert output.shape == input_shape
    assert output.dtype == x.dtype
    # Since fn is identity, output should be same as input
    assert jnp.allclose(output, x)

def test_spatial_linear_attention(rngs):
    """Tests the SpatialLinearAttention module."""
    dim = 16
    heads = 4
    D = 8 # Dimension per head
    b, f, h, w = 2, 3, 4, 5 # Batch, frames, height, width
    input_shape = (b, f, h, w, dim) # Channels last for nnx.Conv
    key = jax.random.PRNGKey(4)
    x = jax.random.normal(key, input_shape)

    # Initialize SpatialLinearAttention module
    attention_module = modules.SpatialLinearAttention(dim=dim, heads=heads, D=D, rngs=rngs)

    # Run the forward pass
    output = attention_module(x)

    # Assertions
    assert output.shape == input_shape
    assert output.dtype == x.dtype 

def test_pre_norm(rngs):
    """Tests the PreNorm module."""
    dim = 10
    batch_size = 4
    input_shape = (batch_size, dim)
    key = jax.random.PRNGKey(5)
    x = jax.random.normal(key, input_shape)

    # Simple function (e.g., identity)
    def identity_fn(inp):
      return inp

    # Initialize PreNorm module
    pre_norm_module = modules.PreNorm(dim=dim, fn=identity_fn, rngs=rngs)

    # Run the forward pass
    output = pre_norm_module(x)

    # Assertions
    assert output.shape == input_shape
    # PreNorm applies LayerNorm then fn. LayerNorm doesn't guarantee same output as input
    # Check dtype and shape is sufficient for this test
    assert output.dtype == x.dtype

def test_block(rngs):
    """Tests the Block module."""
    in_features = 16
    out_features = 32
    groups = 8
    b, f, h, w = 2, 3, 10, 10 # Batch, frames, height, width
    # nnx.Conv expects channels last: (N, ..., C)
    # nnx.GroupNorm expects (N, H, W, C)
    # Let's use a shape compatible with GroupNorm for simplicity in this test
    # (batch, height, width, channels)
    input_shape_gn = (b, h, w, in_features)
    key = jax.random.PRNGKey(6)
    x = jax.random.normal(key, input_shape_gn)

    # Initialize Block module
    block_module = modules.Block(in_features=in_features, 
                                 out_features=out_features,
                                 rngs=rngs, 
                                 groups=groups)

    # Run the forward pass. Pass default scale_shift=None
    output = block_module(x)

    # If it runs, check output shape based on Conv and activation
    # Conv kernel (1,3,3) with default stride/padding might change H, W slightly,
    # but nnx.Conv defaults usually preserve spatial dims if possible (like padding='SAME').
    # Assuming padding='SAME' behavior for the test
    expected_output_shape = (b, h, w, out_features)
    assert output.shape == expected_output_shape
    assert output.dtype == x.dtype
    
    # Optionally, test with scale_shift provided
    scale = jnp.ones((1, 1, 1, out_features)) # Example scale
    shift = jnp.zeros((1, 1, 1, out_features)) # Example shift
    output_with_ss = block_module(x, scale_shift=(scale, shift))
    assert output_with_ss.shape == expected_output_shape
    assert output_with_ss.dtype == x.dtype 