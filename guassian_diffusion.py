import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape
import random as r
from video_diffusion_pytorch.text import tokenize, bert_embed

# Assuming utils.py contains relevant functions like cosine_beta_schedule, extract, etc.
from utils import (
    cosine_beta_schedule,
    extract,
    normalize_img,
    unnormalize_img,
    is_list_str,
    exists,
)

class GaussianDiffusion(nnx.Module):
  """Implements the Gaussian Diffusion process using JAX and Flax NNX.

  Handles the forward diffusion process (adding noise) and the reverse
  process (sampling/denoising) using a provided denoising model (U-Net).
  Also calculates the loss for training the denoising model.

  Attributes:
    denoise_fn: The neural network model (e.g., a U-Net) used to predict noise.
    image_size: The spatial dimensions (height/width) of the images.
    num_frames: The number of frames in the video sequence.
    channels: The number of color channels in the images (e.g., 3 for RGB).
    timesteps: The total number of diffusion timesteps.
    loss_type: The type of loss function ('l1' or 'l2') for training.
    text_use_bert_cls: Whether to use the BERT CLS token for text conditioning.
    use_dynamic_thres: Whether to use dynamic thresholding during sampling.
    dynamic_thres_percentile: Percentile for dynamic thresholding.
    alphas_cumprod: Cumulative product of (1 - betas).
    sqrt_alphas_cumprod: Square root of alphas_cumprod.
    sqrt_one_minus_alphas_cumprod: Square root of (1 - alphas_cumprod).
    log_one_minus_alphas_cumprod: Logarithm of (1 - alphas_cumprod).
    sqrt_recip_alphas_cumprod: Square root of the reciprocal of alphas_cumprod.
    sqrt_recipm1_alphas_cumprod: Square root of (1/alphas_cumprod - 1).
    posterior_variance: Variance of the posterior distribution q(x_{t-1} | x_t, x_0).
    posterior_log_variance_clipped: Clipped log variance of the posterior.
    posterior_mean_coef1: Coefficient 1 for the posterior mean calculation.
    posterior_mean_coef2: Coefficient 2 for the posterior mean calculation.
  """
  def __init__(
      self,
      denoise_fn,                       # Unet model
      *,
      image_size: int,
      num_frames: int,
      text_use_bert_cls: bool = False,
      channels: int = 3,
      timesteps: int = 1000,
      loss_type: str = 'l1',                 # 'l1' or 'l2'
      use_dynamic_thres: bool = False,        # Imagen paper dynamic thresholding
      dynamic_thres_percentile: float = 0.9
  ):
    super().__init__() # Call parent constructor if inheriting from Module directly
    self.channels = channels
    self.image_size = image_size
    self.num_frames = num_frames
    self.denoise_fn = denoise_fn # Ensure this is an nnx.Module or callable
    self.loss_type = loss_type
    self.text_use_bert_cls = text_use_bert_cls
    self.use_dynamic_thres = use_dynamic_thres
    self.dynamic_thres_percentile = dynamic_thres_percentile
    self.num_timesteps = int(timesteps)

    # Calculate diffusion schedule
    betas = cosine_beta_schedule(self.num_timesteps)
    betas = betas.astype(jnp.float32)
    alphas = 1. - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], (1, 0), constant_values=1.0)

    # Precompute values for diffusion q(x_t | x_0)
    self.alphas_cumprod = nnx.Variable(alphas_cumprod)
    self.sqrt_alphas_cumprod = nnx.Variable(jnp.sqrt(alphas_cumprod))
    self.sqrt_one_minus_alphas_cumprod = nnx.Variable(jnp.sqrt(1. - alphas_cumprod))
    self.log_one_minus_alphas_cumprod = nnx.Variable(jnp.log(1. - alphas_cumprod))
    self.sqrt_recip_alphas_cumprod = nnx.Variable(jnp.sqrt(1. / alphas_cumprod))
    self.sqrt_recipm1_alphas_cumprod = nnx.Variable(jnp.sqrt(1. / alphas_cumprod - 1))

    # Precompute values for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    self.posterior_variance = nnx.Variable(posterior_variance)
    # Clip log variance for stability
    self.posterior_log_variance_clipped = nnx.Variable(jnp.log(jnp.maximum(posterior_variance, 1e-20)))
    self.posterior_mean_coef1 = nnx.Variable(betas * jnp.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
    self.posterior_mean_coef2 = nnx.Variable((1. - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1. - alphas_cumprod))


  def q_mean_variance(self, x_start: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calculates the mean and variance of the forward diffusion process q(x_t | x_0).

    Args:
      x_start: The initial image (x_0) with shape (batch, channels, frames, height, width).
      t: The timestep indices (scalar or batch) with shape (batch,).

    Returns:
      A tuple (mean, variance, log_variance):
        mean: Mean of q(x_t | x_0).
        variance: Variance of q(x_t | x_0).
        log_variance: Log variance of q(x_t | x_0).
    """
    mean = extract(self.sqrt_alphas_cumprod.value, t, x_start.shape) * x_start
    variance = extract(1. - self.alphas_cumprod.value, t, x_start.shape)
    log_variance = extract(self.log_one_minus_alphas_cumprod.value, t, x_start.shape)
    return mean, variance, log_variance


  def predict_start_from_noise(self, x_t: jax.Array, t: jax.Array, noise: jax.Array) -> jax.Array:
    """Predicts the initial image (x_0) from the noisy image (x_t) and noise.

    Uses the formula: x_0 = (x_t - sqrt(1 - alphas_cumprod_t) * noise) / sqrt(alphas_cumprod_t)

    Args:
      x_t: The noisy image at timestep t with shape (batch, channels, frames, height, width).
      t: The timestep indices (scalar or batch) with shape (batch,).
      noise: The noise predicted or added at timestep t, same shape as x_t.

    Returns:
      The predicted initial image (x_0) with the same shape as x_t.
    """
    return (
        extract(self.sqrt_recip_alphas_cumprod.value, t, x_t.shape) * x_t -
        extract(self.sqrt_recipm1_alphas_cumprod.value, t, x_t.shape) * noise
    )


  def q_posterior(self, x_start: jax.Array, x_t: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calculates the mean, variance, and log variance of the posterior q(x_{t-1} | x_t, x_0).

    Args:
      x_start: The predicted initial image (x_0) with shape (batch, channels, frames, height, width).
      x_t: The noisy image at timestep t, same shape as x_start.
      t: The timestep indices (scalar or batch) with shape (batch,).

      Returns:
        A tuple (posterior_mean, posterior_variance, posterior_log_variance_clipped):
          posterior_mean: Mean of the posterior distribution.
          posterior_variance: Variance of the posterior distribution.
          posterior_log_variance_clipped: Clipped log variance of the posterior distribution.
      """
    posterior_mean = (
        extract(self.posterior_mean_coef1.value, t, x_t.shape) * x_start +
        extract(self.posterior_mean_coef2.value, t, x_t.shape) * x_t
    )
    posterior_variance = extract(self.posterior_variance.value, t, x_t.shape)
    posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped.value, t, x_t.shape)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped


  def p_mean_variance(self, x: jax.Array, t: jax.Array, clip_denoised: bool, cond = None, cond_scale: float = 1.) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calculates the mean and variance of the reverse process p(x_{t-1} | x_t).

    This involves predicting the noise using the denoise_fn, estimating x_0,
    and then using the formula for the posterior q(x_{t-1} | x_t, x_0_pred)
    to get the mean and variance of the reverse step.

    Args:
      x: The noisy image at timestep t (x_t) with shape (batch, channels, frames, height, width).
      t: The timestep indices (scalar or batch) with shape (batch,).
      clip_denoised: Whether to clip the predicted x_0 to [-1, 1] or use dynamic thresholding.
      cond: Optional conditioning information (e.g., text embeddings).
      cond_scale: Scale factor for classifier-free guidance.

    Returns:
      A tuple (model_mean, model_variance, model_log_variance):
        model_mean: Mean of the distribution p(x_{t-1} | x_t).
        model_variance: Variance of the distribution p(x_{t-1} | x_t).
        model_log_variance: Log variance of the distribution p(x_{t-1} | x_t).
    """
    # Predict the noise using the conditional denoising model
    # NOTE: Assumes denoise_fn handles cond_scale via forward_with_cond_scale method
    # NOTE: Assumes denoise_fn outputs in 'b f h w c' format
    denoise_fn_output = self.denoise_fn.forward_with_cond_scale(
        x,
        t,
        cond=cond,
        cond_scale=cond_scale
    )

    # Reshape predicted noise from 'b f h w c' to match image format 'b c f h w'
    predicted_noise = rearrange(denoise_fn_output, 'b f h w c -> b c f h w')

    # Predict x_0 (the potentially denoised image) using the predicted noise
    x_recon = self.predict_start_from_noise(x, t=t, noise=predicted_noise)

    # Apply clipping or dynamic thresholding to the predicted x_0
    if clip_denoised:
        s = 1.0 # Default threshold for standard clipping
        if self.use_dynamic_thres:
            # Calculate dynamic thresholding percentile (from Imagen)
            # Flatten across all non-batch dimensions to get per-image values
            abs_flat = jnp.abs(rearrange(x_recon, 'b ... -> b (...)'))
            s = jnp.quantile(
                abs_flat,
                self.dynamic_thres_percentile,
                axis=-1
            )
            s = jnp.maximum(s, 1.0) # Ensure threshold is at least 1.0
            # Reshape threshold s for broadcasting: (b,) -> (b, 1, 1, 1, 1)
            # Using x_recon.ndim instead of the previous arr.ndim
            s = jnp.reshape(s, (-1, *(1,) * (x_recon.ndim - 1)))

        # Clip x_recon to [-s, s] and rescale to [-1, 1]
        x_recon = jnp.clip(x_recon, -s, s) / s

    # Calculate the posterior mean and variance using the (potentially clipped) x_recon
    # q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mean, var)
    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
        x_start=x_recon, x_t=x, t=t
    )

    return model_mean, posterior_variance, posterior_log_variance


  def p_sample(self, x: jax.Array, t: jax.Array, cond=None, cond_scale: float = 1., clip_denoised: bool = True) -> jax.Array:
    """Samples x_{t-1} from the reverse process distribution p(x_{t-1} | x_t).

    Args:
      x: The noisy image at timestep t (x_t) with shape (batch, channels, frames, height, width).
      t: The timestep indices (scalar or batch) with shape (batch,).
      cond: Optional conditioning information.
      cond_scale: Scale factor for classifier-free guidance.
      clip_denoised: Whether to clip the predicted x_0 during mean calculation.

    Returns:
      The sampled image at timestep t-1 (x_{t-1}) with the same shape as x.
    """
    model_mean, _, model_log_variance = self.p_mean_variance(
        x=x,
        t=t,
        clip_denoised=clip_denoised,
        cond=cond,
        cond_scale=cond_scale # Pass cond_scale if p_mean_variance uses it
    )

    # Generate noise for sampling step
    noise = jax.random.normal(jax.random.PRNGKey(r.randint(0, 100)), shape=x.shape, dtype=x.dtype)

    # No noise added at the final step (t=0)
    nonzero_mask = (1.0 - (t == 0).astype(jnp.float32))
    # Reshape mask for broadcasting: (b,) -> (b, 1, 1, 1, 1)
    nonzero_mask = rearrange(nonzero_mask, 'b -> b 1 1 1 1')

    # Combine mean and noise (scaled by variance)
    return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise


  def p_sample_loop(self, shape: tuple, cond=None, cond_scale: float = 1.) -> jax.Array:
    """Generates samples by iteratively applying the reverse diffusion process.

    Starts from random noise and denoises it over `num_timesteps`.

    Args:
      shape: The desired shape of the output samples (batch, channels, frames, height, width).
      cond: Optional conditioning information.
      cond_scale: Scale factor for classifier-free guidance.

    Returns:
      The generated samples (images/videos) with the specified shape, unnormalized to [0, 1].
    """
    batch_size = shape[0]
    # Start with random noise
    img = jax.random.normal(key=jax.random.PRNGKey(r.randint(0, 100)), shape=shape)

    # Iteratively denoise from t=num_timesteps-1 down to t=0
    for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling loop', total=self.num_timesteps):
        current_t = jnp.full((batch_size,), i, dtype=jnp.int32)
        img = self.p_sample(
            img,
            current_t,
            cond=cond,
            cond_scale=cond_scale
            # clip_denoised is True by default in p_sample
        )

    # Unnormalize the final image from [-1, 1] to [0, 1]
    return unnormalize_img(img)


  def sample(self, cond=None, cond_scale: float = 1., batch_size: int = 16) -> jax.Array:
    """Generates samples from the diffusion model.

    Handles potential text conditioning preprocessing before calling p_sample_loop.

    Args:
      cond: Optional conditioning information. If list of strings, they are tokenized and embedded.
      cond_scale: Scale factor for classifier-free guidance.
      batch_size: The number of samples to generate if no condition is provided.
                  If cond is provided, batch_size is inferred from it.

    Returns:
      The generated samples with shape (batch_size, channels, num_frames, image_size, image_size).
    """
    # Preprocess text condition if necessary
    if is_list_str(cond):
        # Assuming bert_embed and tokenize are available and handle lists of strings
        cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)

    # Determine batch size based on condition or argument
    if exists(cond):
        batch_size = cond.shape[0]

    # Define the shape for sampling
    sample_shape = (
        batch_size,
        self.channels,
        self.num_frames,
        self.image_size,
        self.image_size
    )

    # Run the sampling loop
    return self.p_sample_loop(shape=sample_shape, cond=cond, cond_scale=cond_scale)


  def interpolate(self, x1: jax.Array, x2: jax.Array, t: int | None = None, lam: float = 0.5) -> jax.Array:
    """Interpolates between two images x1 and x2 using diffusion.

    Diffuses both images to timestep t, linearly interpolates in the latent space,
    and then denoises the result back to an image.

    Args:
      x1: The first image with shape (batch, channels, frames, height, width).
      x2: The second image, same shape as x1.
      t: The intermediate timestep to diffuse to. Defaults to num_timesteps - 1.
      lam: Interpolation factor (0 <= lam <= 1). lam=0 yields x1, lam=1 yields x2.

    Returns:
      The interpolated image with the same shape as x1 and x2.
    """
    batch_size = x1.shape[0]
    # Default to the last timestep if not provided
    t = t if exists(t) else self.num_timesteps - 1

    assert x1.shape == x2.shape, "Input images must have the same shape for interpolation."
    assert 0.0 <= lam <= 1.0, "Interpolation factor lambda must be between 0 and 1."

    # Create batched timestep array
    t_batched = jnp.full((batch_size,), t, dtype=jnp.int32)

    # Diffuse both images to timestep t
    xt1 = self.q_sample(x1, t=t_batched, noise=None) # Use None to generate noise
    xt2 = self.q_sample(x2, t=t_batched, noise=None)

    # Linearly interpolate in the latent space
    img_latent = (1 - lam) * xt1 + lam * xt2

    # Denoise the interpolated latent image from t down to 0
    for i in tqdm(reversed(range(0, t)), desc='Interpolation sampling', total=t):
        current_t = jnp.full((batch_size,), i, dtype=jnp.int32)
        # Note: Interpolation typically doesn't use conditioning
        img_latent = self.p_sample(img_latent, current_t)

    return img_latent # Return the final denoised image


  def q_sample(self, x_start: jax.Array, t: int, noise: jax.Array | None):
    """Samples from the forward diffusion process q(x_t | x_0) at timestep t.

    Adds noise to the initial image x_start according to the diffusion schedule.

    Args:
      x_start: The initial image (x_0) with shape (batch, channels, frames, height, width).
      t: The timestep indices (scalar or batch) with shape (batch,).
      noise: Optional noise tensor to use. If None, random noise is generated.
             Must have the same shape as x_start.

    Returns:
      The noisy image x_t with the same shape as x_start.
    """
    # Generate noise if not provided
    if noise is None:
        noise = jax.random.normal(jax.random.PRNGKey(r.randint(0, 100)), shape=x_start.shape)

    # Apply the forward diffusion formula
    return (
        extract(self.sqrt_alphas_cumprod.value, t, x_start.shape) * x_start +
        extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_start.shape) * noise
    )


  def p_losses(self, x_start: jax.Array, t: jax.Array, cond=None, noise: jax.Array | None = None, **kwargs) -> jax.Array:
    """Calculates the training loss for the diffusion model.

    Adds noise to x_start to get x_t, predicts the noise using the denoise_fn,
    and computes the loss (L1 or L2) between the predicted and actual noise.

    Args:
      x_start: The initial images (batch, channels, frames, height, width).
      t: The timestep indices (scalar or batch) with shape (batch,).
      cond: Optional conditioning information.
      noise: Optional noise tensor to use for q_sample. If None, random noise is generated.
      **kwargs: Additional keyword arguments passed to the denoise_fn.

    Returns:
      The calculated loss (scalar).
    """
    # Generate noise if not provided
    if noise is None:
        noise = jax.random.normal(jax.random.PRNGKey(r.randint(0, 100)), shape=x_start.shape)

    # Create noisy version of x_start at timestep t
    # TBD comment clarification: This correctly samples x_t using the formula for q(x_t|x_0)
    x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

    # Preprocess text condition if necessary
    if is_list_str(cond):
        cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)

    # Predict the noise added to x_noisy using the U-Net model
    # Assuming denoise_fn outputs in 'b f h w c' format
    denoise_fn_output = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

    # Reshape predicted noise to match the target noise shape 'b c f h w'
    # TBD: Check if this is correct
    predicted_noise = rearrange(denoise_fn_output, 'b f h w c -> b c f h w')

    # Calculate the loss
    if self.loss_type == 'l1':
        loss = jnp.mean(jnp.abs(predicted_noise - noise))
    elif self.loss_type == 'l2':
        loss = jnp.mean((predicted_noise - noise)**2)
    else:
        raise ValueError(f"Unsupported loss type: {self.loss_type}")

    return loss


  def __call__(self, x: jax.Array, *args, **kwargs) -> jax.Array:
    """Defines the forward pass for training, calculating the loss.

    Normalizes the input image, selects random timesteps, and calls p_losses.

    Args:
      x: The initial images (batch, channels, frames, height, width).
      *args: Positional arguments passed to p_losses (e.g., conditioning).
      **kwargs: Keyword arguments passed to p_losses.

    Returns:
      The calculated training loss (scalar).
    """
    batch_size = x.shape[0]
    img_size = self.image_size
    # Check input shape matches configuration
    check_shape(x, 'b c f h w', b=batch_size, c=self.channels, f=self.num_frames, h=img_size, w=img_size)

    # Sample random timesteps for the batch
    t = jax.random.randint(jax.random.PRNGKey(r.randint(0, 100)), (batch_size,), 0, self.num_timesteps, dtype=jnp.int32)

    # Normalize images from [0, 1] to [-1, 1] (if needed)
    x_normalized = normalize_img(x)

    # Calculate and return the loss
    return self.p_losses(x_normalized, t, *args, **kwargs)