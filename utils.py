import jax
import jax.numpy as jnp
import random


def exists(x):
  """Check if a value exists (is not None).
  
  Args:
    x: The value to check
    
  Returns:
    bool: True if x is not None, False otherwise
  """
  return x is not None

def noop(*args, **kwargs):
  """A no-operation function that does nothing.
  
  Args:
    *args: Any positional arguments (ignored)
    **kwargs: Any keyword arguments (ignored)
  """
  pass

def is_odd(n):
  """Check if a number is odd.
  
  Args:
    n (int): The number to check
    
  Returns:
    bool: True if n is odd, False otherwise
  """
  return (n % 2) == 1

def default(val, d):
  """Return a default value if the input is None.
  
  Args:
    val: The value to check
    d: The default value to return if val is None. Can be a callable or a value.
    
  Returns:
    The value if it exists, otherwise the default value
  """
  if exists(val):
    return val

  return d() if callable(d) else d

def cycle(dl):
  """Create an infinite iterator that cycles through a data loader.
  
  Args:
    dl: The data loader to cycle through
    
  Yields:
    The next item from the data loader
  """
  while True:
    for data in dl:
      yield data

def prob_mask_like(shape, prob):
  """Generate a boolean mask with given probability.
  
  Args:
    shape: The shape of the output mask
    prob: The probability of True values in the mask (between 0 and 1)
    
  Returns:
    jnp.ndarray: A boolean mask of the specified shape where each element is True
                 with probability prob
  """
  if prob == 1:
    return jnp.ones(shape=shape, dtype=jnp.bool)
  elif prob == 0:
    return jnp.zeros(shape=shape, dtype=jnp.bool)
  else:
    return jax.random.uniform(key = jax.random.PRNGKey(random.randint(0, 100)), shape = shape, minval = 0, maxval=1) < prob