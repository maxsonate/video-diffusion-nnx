import jax
import jax.numpy as jnp
import random
from flax import nnx
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
from functools import partial
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, args as ocp_args
import logging
from typing import Any
from jax import tree_util
# PyTree is a JAX pytree (arbitrary nested structure), use Any for compatibility
PyTree = Any


CHANNELS_TO_MODE = {
    1:'L',
    2:'RGB',
    3:'RGBA'
}


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
      l2_norm: L2 norm of the pre clipped gradients

  Returns:
      Normalized gradients
  """
  # Compute squared L2 norm per leaf
  grad_squared = tree_util.tree_map(lambda x: jnp.sum(x**2), grads)
  # Sum over all leaves to get total squared norm
  total_sq = tree_util.tree_reduce(jnp.add, grad_squared, 0.0)
  # Global L2 norm
  l2_norm = jnp.sqrt(total_sq + epsilon)

  # logging.info(f'l2 norm:{l2_norm}') # Keep as comment or remove if not needed for debug

  # Compute clipping scale; ensure norm does not exceed max_grad_norm
  scale = jnp.minimum(max_grad_norm / (l2_norm + epsilon), 1.0)
  # logging.info(f'clip scale:{scale}') # Keep as comment or remove if not needed for debug
  return tree_util.tree_map(lambda x: x * scale, grads), l2_norm



def clip_grad_norm_with_tb_logging(grads, max_grad_norm, tb_writer, step, epsilon=1e-6):
  """Clips gradients by global L2 norm and logs diagnostic information to TensorBoard.

  Computes the global L2 norm of all gradients in the PyTree `grads`.
  If the norm exceeds `max_grad_norm`, scales all gradients uniformly such that
  the global norm equals `max_grad_norm`. Otherwise, gradients remain unchanged.

  Additionally, logs the following to TensorBoard at the given `step`:
    - 'global_grad_norm/pre_clip': The global L2 norm before clipping.
    - 'global_grad_norm/clip_scale': The scaling factor applied (1.0 if no clipping occurred).
    - 'per_param_grad_norm/<path>': The L2 norm of the gradient for each individual
      parameter (leaf node) in the `grads` PyTree. The `<path>` reflects the
      parameter's position within the PyTree structure (e.g., "['layer1'][0]['weights']").

  Args:
    grads: A PyTree of gradients.
    max_grad_norm: The maximum allowed global L2 norm for the gradients.
    tb_writer: A TensorBoard summary writer instance (e.g., from `flax.metrics.tensorboard`).
    step: The current training step (integer) for logging.
    epsilon: A small float added to the norm calculation for numerical stability,
      preventing division by zero.

  Returns:
    A tuple containing:
      - grads: The (potentially scaled) PyTree of gradients.
      - l2_norm: The computed global L2 norm *before* clipping.
  """

  # Compute squared L2 norm per leaf
  grad_squared = tree_util.tree_map(lambda x: jnp.sum(x**2), grads)
  # Sum over all leaves to get total squared norm
  total_sq = tree_util.tree_reduce(jnp.add, grad_squared, 0.0)
  # Global L2 norm (pre-clipping)
  l2_norm = jnp.sqrt(total_sq + epsilon)

  # Compute clipping scale; ensure norm does not exceed max_grad_norm
  scale = jnp.minimum(max_grad_norm / (l2_norm + epsilon), 1.0)

  # Log global norm and scale factor
  tb_writer.add_scalar('global_grad_norm/pre_clip', np.asarray(l2_norm), step)
  tb_writer.add_scalar('global_grad_norm/clip_scale', np.asarray(scale), step)

  # Apply clipping
  clipped_grads = tree_util.tree_map(lambda x: x * scale, grads)

  # Log per-parameter norms to TensorBoard (using the squared norms calculated earlier)
  per_param_norms = tree_util.tree_map(lambda sq: jnp.sqrt(sq + epsilon), grad_squared)
  per_param_norms_with_paths = tree_util.tree_leaves_with_path(per_param_norms)

  # Format path and log each per-parameter norm
  for path, leaf_norm in per_param_norms_with_paths:
    path_str_parts = []
    for entry in path:
        if isinstance(entry, tree_util.DictKey):
            # Use '.' for nesting instead of brackets for cleaner TB names
            path_str_parts.append(f"{entry.key}")
        elif isinstance(entry, tree_util.SequenceKey):
            path_str_parts.append(f"{entry.idx}")
        else:
            path_str_parts.append(str(entry))
    # Join with '.' for a hierarchical tag in TensorBoard
    path_str = '.'.join(path_str_parts)
    # Add a prefix to avoid potential collisions with other logs
    tb_writer.add_scalar(f'per_param_grad_norm/{path_str}', np.asarray(leaf_norm), step)

  return clipped_grads, l2_norm



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
  """Unnormalizes an image tensor from [-1, 1] back to [0, 1].
  
  Args:
    t: The input tensor, assumed to be in the range [-1, 1].

  Returns:
    jnp.ndarray: The unnormalized tensor in the range [0, 1].
  """
  return (t + 1) * 0.5

# Function that scales the pixel values of an image to the range [-1, 1].
def normalize_img(t):
  """Normalizes an image tensor from [0, 1] to the range [-1, 1].
  
  Args:
    t: The input tensor, assumed to be in the range [0, 1].

  Returns:
    jnp.ndarray: The normalized tensor in the range [-1, 1].
  """
  return t * 2 - 1

def is_list_str(x):
  """Checks if the input is a list or tuple containing only strings.
  
  Args:
    x: The input variable to check.

  Returns:
    bool: True if x is a list or tuple containing only strings, False otherwise.
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

# The function takes an image object and the number of channels as inputs and converts each frame to the specified model before yielding it.
def seek_all_images(img, channels = 3):
  """Yields all frames from a PIL Image object, converted to the specified mode.

  Args:
    img (PIL.Image.Image): The input image object (e.g., loaded from a GIF).
    channels (int): The desired number of channels (1: 'L', 3: 'RGB', 4: 'RGBA'). Defaults to 3.

  Yields:
    PIL.Image.Image: Each frame of the image, converted to the specified mode.
  
  Raises:
    AssertionError: If the number of channels is invalid.
  """
  assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
  mode = CHANNELS_TO_MODE[channels]
  i = 0
  while True:
    try:
      img.seek(i)
      yield img.convert(mode)
    except EOFError:
      break
    i += 1


# Function that takes a tensor of shape (channels, frames, height, width) representing
# a sequence of frames and saves them as GIF file at the specified path
# Tensor of shape (frames, height, width, channels) -> gif
def video_array_to_gif(arr, path, duration = 120, loop = 0, optimize = True):
  """Converts a video tensor to a GIF file.

  NOTE: This function currently has a potential bug in how it processes frames.
  It splits the array by channels instead of frames. See implementation notes.
  It also depends on numpy, torch, torchvision, and PIL.

  Args:
    arr (np.ndarray): Input tensor of shape (channels, frames, height, width).
                      Should contain pixel values suitable for image conversion.
    path (str): The output path for the GIF file.
    duration (int): Duration (in milliseconds) for each frame. Defaults to 120.
    loop (int): Number of times the GIF should loop (0 for infinite). Defaults to 0.
    optimize (bool): Whether to optimize the GIF. Defaults to True.

  Returns:
    list[PIL.Image.Image]: A list of PIL Image objects representing the frames.
  """

  images_arr = np.split(arr, arr.shape[0], axis=0) 
  images_arr = list(map(partial(np.squeeze, axis=0), images_arr)) # Squeeze might fail if channels != 1
  # logging.info(images_arr[0].shape, len(images_arr)) # Keep as comment or remove if not needed for debug
  images = map(T.ToPILImage(), images_arr) # Assumes T.ToPILImage handles the squeezed shape
  first_img, *rest_imgs = images
  first_img.save(path,
                 save_all = True,
                 append_images = rest_imgs,
                 duration = duration,
                 loop = loop,
                 optimize = optimize)
  return images

# Function that pads or truncates an array of shape (channels,frames, height, width)
# to the specific number of frames
# If the array already has the specified number of frames, the function returns it unchanged.
# If the array has more frames, the function truncates it to the specified number of frames.
# If the array has fewer frames, the function pads it with zeros to the specified number of frames.
def cast_num_frames(t, *, frames):
  """Pads or truncates a video tensor along the frame dimension.

  Args:
    t (jnp.ndarray): Input tensor of shape (channels, frames, height, width).
    frames (int): The target number of frames.

  Returns:
    jnp.ndarray: The tensor adjusted to have the specified number of frames.
                 Padding uses zeros.
  """
  num_frames = t.shape[1]
  if num_frames == frames:
    return t
  elif num_frames > frames:
    return t[:,:frames, ...]
  else:
    return jnp.pad(t, pad_width=((0,0), (0, frames - num_frames), (0, 0), (0, 0)))


# Function used to extract a text description of each GIF for use as a condition.
def get_text_from_path(path):
    """Extracts a descriptive text string from a file path.

    Assumes the filename (without extension) contains hyphen or underscore-separated words.

    Args:
      path (str): The file path.

    Returns:
      str: A space-separated string derived from the filename.
    """
    out = path.split('/')[-1]
    out = out.split('.')[0]
    out = out.replace('-', ' ')
    out = out.replace('_', ' ')
    return out

def identity(t, *args, **kwargs):
  """Identity function. Returns the first argument unchanged.

  Args:
    t: The input value.
    *args: Additional positional arguments (ignored).
    **kwargs: Additional keyword arguments (ignored).

  Returns:
    The input value `t`.
  """
  return t


def save_checkpoint(ckpt_manager: CheckpointManager,model_params: PyTree, ema_params: PyTree, step: int):
    """Saves the state of an NNX model using Orbax CheckpointManager.

    Args:
        ckpt_manager: The Orbax CheckpointManager instance to use.
        model: The Flax NNX model instance to save.
        ema_params: The Flax NNX model instance to save.
        step (int): The current training step, used as the checkpoint identifier.
    """
    # # Get the model state dictionar
    # _, state = nnx.split(model)

    # Combine states into one PyTree dict
    save_items = {
        'model': model_params,
        'ema_params': ema_params,
    }

    # Save the checkpoint as a single unnamed PyTree
    ckpt_manager.save(
        step,
        args=ocp_args.StandardSave(save_items),  # Save combined PyTree dict
        force=True  # Allow overwriting if step already exists
    )
    ckpt_manager.wait_until_finished()

    logging.info(f"Checkpoint saved at step {step}")

def load_checkpoint(model: nnx.Module, step: int, path: str, ckpt_manager: CheckpointManager | None = None, load_ema_params: bool = False) -> nnx.Module:
    """Loads the state of an NNX model from an Orbax checkpoint.

    Note: This function modifies the input model object by merging the loaded state.
    However, due to how NNX works, it's recommended to use the returned model object.

    Args:
        model: The Flax NNX model instance with the correct structure but potentially
               uninitialized or incorrect parameters.
        step (int): The training step of the checkpoint to load.
        path (str): The base directory path containing the checkpoint subdirectory
                    (e.g., '{path}/{step}/state').
        ckpt_manager (CheckpointManager | None, optional): The Orbax CheckpointManager instance
                                                          to use for loading the checkpoint.
                                                          If None, a new Checkpointer will be created.
                                                          Defaults to None.
        load_ema_params (bool, optional): Whether to load EMA parameters. Defaults to False.
    Returns:
        The NNX model instance with the loaded state merged into it.
    """
    # Create an Orbax checkpointer
    if ckpt_manager is None:
        ckpt_manager = ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions())
    #checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    # Split out the abstract model state
    graphdef, abstract_state = nnx.split(model)
    # Create an abstract template for both model and EMA (same shape)
    abstract = {
        'model': abstract_state,
        'ema_params': abstract_state
    }
    # Restore the combined dict of model and EMA states
    state_dict = ckpt_manager.restore(
        step,
        args=ocp_args.StandardRestore(abstract)
    )
    model_params = state_dict['model']
    ema_params = state_dict['ema_params']

    # Apply the state dictionary to the model
    if load_ema_params:
        model = nnx.merge(graphdef, ema_params)
        logging.info("Loaded EMA parameters")
    else:
        model = nnx.merge(graphdef, model_params)
        logging.info("Loaded model parameters")
    logging.info(f"Checkpoint loaded from step: {step}")
    return model, ema_params