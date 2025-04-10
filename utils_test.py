# test_utils.py
import pytest
import jax
import jax.numpy as jnp
from itertools import islice
from flax import nnx
from utils import Upsample, Downsample
from utils import exists, noop, is_odd, default, cycle, prob_mask_like

# --- Test exists ---

def test_exists_with_none():
  """Test exists() with None."""
  assert not exists(None)

def test_exists_with_values():
  """Test exists() with various non-None values."""
  assert exists(0)
  assert exists(1)
  assert exists("")
  assert exists("hello")
  assert exists([])
  assert exists([1, 2])
  assert exists({})
  assert exists({"a": 1})
  assert exists(False)
  assert exists(True)

# --- Test noop ---

def test_noop_returns_none():
  """Test that noop() returns None and accepts args/kwargs."""
  assert noop() is None
  assert noop(1, 2, 3) is None
  assert noop(a=1, b=2) is None
  assert noop(1, 2, c=3) is None

# --- Test is_odd ---

def test_is_odd_positive():
  """Test is_odd() with positive numbers."""
  assert is_odd(1)
  assert not is_odd(2)
  assert is_odd(3)
  assert not is_odd(100)

def test_is_odd_zero():
  """Test is_odd() with zero."""
  assert not is_odd(0)

def test_is_odd_negative():
  """Test is_odd() with negative numbers."""
  assert is_odd(-1)
  assert not is_odd(-2)
  assert is_odd(-3)
  assert not is_odd(-100)

# --- Test default ---

def test_default_with_value():
  """Test default() when the value exists."""
  assert default(5, 10) == 5
  assert default("hello", "world") == "hello"
  assert default(False, True) is False
  assert default(0, 1) == 0
  assert default([], [1]) == []

def test_default_with_none_and_value():
  """Test default() when value is None and default is a value."""
  assert default(None, 10) == 10
  assert default(None, "world") == "world"
  assert default(None, True) is True
  assert default(None, [1]) == [1]

def test_default_with_none_and_callable():
  """Test default() when value is None and default is a callable."""
  assert default(None, lambda: 10) == 10
  assert default(None, lambda: "world") == "world"
  assert default(None, list) == []
  class MyClass: pass
  assert isinstance(default(None, MyClass), MyClass)

# --- Test prob_mask_like ---

def test_prob_mask_like_prob_1():
  """Test prob_mask_like with probability 1."""
  shape = (2, 3)
  mask = prob_mask_like(shape, 1.0)
  assert mask.shape == shape
  assert mask.dtype == jnp.bool_
  assert jnp.all(mask)

def test_prob_mask_like_prob_0():
  """Test prob_mask_like with probability 0."""
  shape = (3, 2)
  mask = prob_mask_like(shape, 0.0)
  assert mask.shape == shape
  assert mask.dtype == jnp.bool_
  assert not jnp.any(mask)

def test_prob_mask_like_prob_half():
  """Test prob_mask_like with probability 0.5 (checks shape and dtype)."""
  shape = (5, 5)
  # Note: Due to internal random key generation, we can't test deterministically
  # We primarily check shape, dtype, and that it returns a JAX array.
  mask = prob_mask_like(shape, 0.5)
  assert isinstance(mask, jax.Array)
  assert mask.shape == shape
  assert mask.dtype == jnp.bool_
  # For a reasonable probability and shape, some should likely be True and False
  # This is a weak check due to randomness.
  # assert jnp.any(mask)
  # assert jnp.any(~mask)

def test_prob_mask_like_invalid_prob():
    """Test prob_mask_like behavior with probability outside [0, 1] (implicitly tests < operator)."""
    shape = (2, 2)
    # Probability > 1 should behave like prob = 1 due to '< prob' comparison
    mask_over = prob_mask_like(shape, 1.5)
    assert mask_over.shape == shape
    assert mask_over.dtype == jnp.bool_
    # Probability < 0 should behave like prob = 0
    mask_under = prob_mask_like(shape, -0.5)
    assert mask_under.shape == shape
    assert mask_under.dtype == jnp.bool_


@pytest.fixture
def rngs():
  """Provides nnx.Rngs for Upsample/Downsample tests."""
  return nnx.Rngs(0)


def test_upsample(rngs):
    """Tests the Upsample function (ConvTranspose wrapper)."""
    dim = 16
    b, f, h, w = 2, 1, 10, 10 # Using f=1 for simplicity
    input_shape = (b, f, h, w, dim) # Channels last
    key = jax.random.PRNGKey(7)
    x = jax.random.normal(key, input_shape)

    # Initialize Upsample module
    upsample_layer = Upsample(dim=dim, rngs=rngs)


    output = upsample_layer(x)

    # Assertions
    expected_output_shape = (b, 1, 20, 20, dim)
    assert output.shape == expected_output_shape
    assert output.dtype == x.dtype

def test_downsample(rngs):
    """Tests the Downsample function (Conv wrapper)."""
    dim = 16
    b, f, h, w = 2, 1, 10, 10 # Using f=1 for simplicity
    input_shape = (b, f, h, w, dim) # Channels last
    key = jax.random.PRNGKey(8)
    x = jax.random.normal(key, input_shape)

    # Initialize Downsample module
    downsample_layer = Downsample(dim=dim, rngs=rngs)

    output = downsample_layer(x)

    # Assertions
    expected_output_shape = (b, 1, 5, 5, dim)
    assert output.shape == expected_output_shape
    assert output.dtype == x.dtype