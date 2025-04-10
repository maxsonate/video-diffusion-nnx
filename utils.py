import jax
import jax.numpy as jnp
import random
from flax import nnx


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
  
def Upsample(dim:int, rngs:nnx.Rngs):
  """Upsamples the input tensor by a factor of 4 using a transposed convolution.
  
  Args:
    dim: The number of input channels
    rngs: The random number generator
    
  Returns:
    nnx.ConvTranspose: A transposed convolution layer that upsamples the input tensor by a factor of 4
  """
  return nnx.ConvTranspose(dim, dim, (1, 4, 4), (1, 2, 2), rngs=rngs)

def Downsample(dim:int, rngs:nnx.Rngs):
  """Downsamples the input tensor by a factor of 4 using a convolution.
  
  Args:
    dim: The number of input channels
    rngs: The random number generator
    
  Returns:
    nnx.Conv: A convolution layer that downsamples the input tensor by a factor of 4
  """
  return nnx.Conv(dim, dim, (1, 4, 4), (1, 2, 2), rngs=rngs)

def clip_grad_norm(grads, max_grad_norm, epsilon=1e-6):
  """
  Enhanced gradient norm clipping with additional safety.

  Args:
      grads: Pytree of gradients
      max_grad_norm: Maximum allowed gradient norm
      epsilon: Small value to prevent division by zero

  Returns:
      Normalized gradients
  """
  grad_squared = jax.tree.map(lambda x: jnp.sum(x**2), grads)
  l2_norm = jnp.sqrt(jax.tree.reduce(jnp.add, grad_squared) + epsilon)
  total_leaves = len(jax.tree.leaves(grad_squared))

  l2_norm = l2_norm / total_leaves
  print(f'l2 norm:{l2_norm}')

  scale = jnp.minimum(max_grad_norm / l2_norm, 1.0)
  print(f'clip scale:{scale}')
  return jax.tree.map(lambda x: x * scale, grads)

grads_test = {
        'weights': jnp.array([1, 2, 3]),
        'biases': jnp.array([4, 5]),
        }



# TODO: Double check this, not too sure about the implementation:
def extract(a:jax.Array, t, x_shape):
  """Extracts elements from a tensor based on the provided indices.
  
  Args:
    a: The input tensor
    t: The indices to extract
    x_shape: The shape of the input tensor

  Returns:
    jax.Array: A tensor containing the extracted elements
  """
  b, *_ = t.shape
  out = jnp.take_along_axis(a, t, axis=-1)
  return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s = 0.008):
  """Generates a cosine beta schedule for the given number of timesteps.
  
  Args:
    timesteps: The number of timesteps in the schedule
    s: A small constant to prevent division by zero

  Returns:
    jnp.ndarray: A tensor containing the beta schedule
  """
  steps = timesteps + 1
  x = jnp.linspace(0, timesteps, steps, dtype = jnp.float64)
  alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
  return jnp.clip(betas, 0, 0.9999)


def unnormalize_img(t):
  """Unnormalizes the input tensor by scaling it to the range [-1, 1].
  
  Args:
    t: The input tensor

  Returns:
    jnp.ndarray: The unnormalized tensor
  """
  return (t + 1) * 0.5

# Function that scales the pixel values of an image to the range [-1, 1].
def normalize_img(t):
  """Normalizes the input tensor to the range [-1, 1].
  
  Args:
    t: The input tensor

  Returns:
    jnp.ndarray: The normalized tensor
  """
  return t * 2 - 1

def is_list_str(x):
  """Checks if the input is a list or tuple and contains only strings.
  
  Args:
    x: The input to check

  Returns:
    bool: True if x is a list or tuple and contains only strings, False otherwise
  """
  if not isinstance(x , (list, tuple)):
    return False
  return all([type(el) == str for el in x])

def num_to_groups(num, divisor):
    """
    Divide a number into groups of a specified size.

    Args:
        num (int): The number to be divided into groups
        divisor (int): The size of each group

    Returns:
        list: A list of group sizes where all groups except possibly the last are of size 'divisor'
    """
    groups, remainder = divmod(num, divisor)
    result = [divisor] * groups

    if remainder:
        result.append(remainder)

    return result