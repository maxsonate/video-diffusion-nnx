import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape
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

  Requires explicit PRNGKeys for all random operations.

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

    Note: This function itself doesn't require a PRNGKey as randomness comes
          from the sampling step (p_sample). Denoise_fn might need one internally
          if it uses dropout, but that should be handled within denoise_fn.

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
            # Using einops is clearer:
            s = rearrange(s, 'b -> b 1 1 1 1')

        # Clip x_recon to [-s, s] and rescale to [-1, 1]
        x_recon = jnp.clip(x_recon, -s, s) / s

    # Calculate the posterior mean and variance using the (potentially clipped) x_recon
    # q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mean, var)
    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
        x_start=x_recon, x_t=x, t=t
    )

    return model_mean, posterior_variance, posterior_log_variance


  def p_sample(self, x: jax.Array, t: jax.Array, key: jax.random.PRNGKey, cond=None, cond_scale: float = 1., clip_denoised: bool = True) -> jax.Array:
    """Samples x_{t-1} from the reverse process distribution p(x_{t-1} | x_t).

    Args:
      x: The noisy image at timestep t (x_t).
      t: The timestep indices (scalar or batch).
      key: JAX PRNGKey for generating sampling noise.
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

    # Generate noise for sampling step using the provided key
    noise = jax.random.normal(key, shape=x.shape, dtype=x.dtype)

    # No noise added at the final step (t=0)
    nonzero_mask = (1.0 - (t == 0).astype(jnp.float32))
    nonzero_mask = rearrange(nonzero_mask, 'b -> b 1 1 1 1') # Reshape mask for broadcasting

    # Combine mean and noise (scaled by variance)
    return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise


  def p_sample_loop(self, shape: tuple, key: jax.random.PRNGKey, cond=None, cond_scale: float = 1.) -> jax.Array:
    """Generates samples by iteratively applying the reverse diffusion process.

    Starts from random noise and denoises it over `num_timesteps`.

    Args:
      shape: The desired shape of the output samples (batch, channels, frames, height, width).
      key: JAX PRNGKey for the entire sampling loop.
      cond: Optional conditioning information.
      cond_scale: Scale factor for classifier-free guidance.

    Returns:
      The generated samples (images/videos) with the specified shape, unnormalized to [0, 1].
    """
    batch_size = shape[0]
    # Split key for initial noise and the loop
    key, init_noise_key = jax.random.split(key)
    img = jax.random.normal(key=init_noise_key, shape=shape)

    # Iteratively denoise from t=num_timesteps-1 down to t=0
    for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling loop', total=self.num_timesteps):
        current_t = jnp.full((batch_size,), i, dtype=jnp.int32)
        # Split key for this step's sample
        key, step_key = jax.random.split(key)
        img = self.p_sample(
            img,
            current_t,
            key=step_key, # Pass the key for this step
            cond=cond,
            cond_scale=cond_scale
            # clip_denoised is True by default in p_sample
        )

    # Unnormalize the final image from [-1, 1] to [0, 1]
    return unnormalize_img(img)


  def sample(self, key: jax.random.PRNGKey, cond=None, cond_scale: float = 1., batch_size: int = 16) -> jax.Array:
    """Generates samples from the diffusion model.

    Handles potential text conditioning preprocessing before calling p_sample_loop.

    Args:
      key: JAX PRNGKey for the sampling process.
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
    return self.p_sample_loop(shape=sample_shape, key=key, cond=cond, cond_scale=cond_scale)


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


  def q_sample(self, x_start: jax.Array, t: int, key: jax.random.PRNGKey, noise: jax.Array | None = None):
    """Samples from the forward diffusion process q(x_t | x_0).

    Args:
      x_start: Initial image (x_0).
      t: Timestep indices.
      key: JAX PRNGKey, required if noise is None.
      noise: Optional noise tensor. If None, random noise is generated using key.

    Returns:
      The noisy image x_t with the same shape as x_start.
    """
    # Generate noise if not provided
    if noise is None:
        assert key is not None, "A PRNGKey must be provided to q_sample if noise is not."
        noise = jax.random.normal(key, shape=x_start.shape)

    return (
        extract(self.sqrt_alphas_cumprod.value, t, x_start.shape) * x_start +
        extract(self.sqrt_one_minus_alphas_cumprod.value, t, x_start.shape) * noise
    )


  def p_losses(self, x_start: jax.Array, t: jax.Array, key: jax.random.PRNGKey, cond=None, noise: jax.Array | None = None, **kwargs) -> jax.Array:
    """Calculates the training loss for the diffusion model.

    Adds noise to x_start to get x_t, predicts the noise using the denoise_fn,
    and computes the loss (L1 or L2) between the predicted and actual noise.

    Args:
      x_start: The initial images (batch, channels, frames, height, width).
      t: The timestep indices (scalar or batch) with shape (batch,).
      key: JAX PRNGKey for noise generation and potentially q_sample.
      cond: Optional conditioning information.
      noise: Optional noise tensor to use for q_sample. If None, random noise is generated.
      **kwargs: Additional keyword arguments passed to the denoise_fn.

    Returns:
      The calculated loss (scalar).
    """
    # Split key for noise generation and potentially q_sample
    key, noise_key, q_sample_key = jax.random.split(key, 3)

    # Generate noise if not provided
    if noise is None:
        noise = jax.random.normal(noise_key, shape=x_start.shape)

    # Create noisy version using q_sample, passing a key if it needs to generate noise
    x_noisy = self.q_sample(x_start=x_start, t=t, key=q_sample_key, noise=noise) # Use generated noise

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


  def __call__(self, x: jax.Array, key: jax.random.PRNGKey, *args, **kwargs) -> jax.Array:
    """Defines the forward pass for training, calculating the loss.

    Normalizes the input image, selects random timesteps, and calls p_losses.

    Args:
      x: The initial images (batch, channels, frames, height, width).
      key: JAX PRNGKey for the entire loss calculation step.
      *args: Positional arguments passed to p_losses (e.g., conditioning).
      **kwargs: Keyword arguments passed to p_losses.

    Returns:
      The calculated training loss (scalar).
    """
    batch_size = x.shape[0]
    img_size = self.image_size
    # Check input shape matches configuration
    check_shape(x, 'b c f h w', b=batch_size, c=self.channels, f=self.num_frames, h=img_size, w=img_size)

    # Split key for timestep sampling and loss calculation
    key, t_key, loss_key = jax.random.split(key, 3)

    # Sample random timesteps for the batch
    t = jax.random.randint(t_key, (batch_size,), 0, self.num_timesteps, dtype=jnp.int32)

    # Normalize images from [0, 1] to [-1, 1] (if needed)
    x_normalized = normalize_img(x)

    # Calculate and return the loss, passing the loss-specific key
    return self.p_losses(x_normalized, t, key=loss_key, *args, **kwargs)