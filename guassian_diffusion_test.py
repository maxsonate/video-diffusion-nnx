import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from unittest.mock import patch, MagicMock

# Import the class to test
from guassian_diffusion import GaussianDiffusion

# Mock the utility functions that might not be available or have complex dependencies
# If these are simple and available in utils.py, this mocking might not be needed
mock_bert_embed = MagicMock(return_value=jnp.zeros((1, 77, 768))) # Example shape
mock_tokenize = MagicMock(return_value=["test"]) # Example output
mock_check_shape = MagicMock() # Doesn't need to return anything, just not error

# Define a mock U-Net (denoise_fn)
class MockDenoiseFn(nnx.Module):
    def __init__(self, channels=3):
        self.channels = channels
        # No real parameters needed for this mock

    def __call__(self, x, t, cond=None, **kwargs):
        # Returns zeros in the expected output format of the *real* U-Net
        # which is assumed to be 'b f h w c' before rearrange
        b, _, f, h, w = x.shape
        return jnp.zeros((b, f, h, w, self.channels), dtype=x.dtype)
    
    # Mock the specific method called in p_mean_variance if different from __call__
    def forward_with_cond_scale(self, x, t, cond=None, cond_scale=1.0, **kwargs):
         b, _, f, h, w = x.shape
         # Return dummy noise of expected shape (b f h w c)
         return jnp.zeros((b, f, h, w, self.channels), dtype=x.dtype)


class TestGaussianDiffusion(unittest.TestCase):

    def setUp(self):
        """Set up a GaussianDiffusion instance for testing."""
        self.image_size = 8 # Small image size for tests
        self.num_frames = 2
        self.channels = 3
        self.timesteps = 10 # Few timesteps for faster tests
        self.batch_size = 2

        # Create the mock denoise function
        # Pass necessary args if your real mock/model needs them
        mock_unet = MockDenoiseFn(channels=self.channels)

        # Instantiate the class under test
        # Patching tqdm to prevent progress bar printing during tests
        with patch('guassian_diffusion.tqdm', lambda x, **kwargs: x): # Simple patch for tqdm
             # Patching the external utility functions directly in the target module's namespace
             with patch('guassian_diffusion.bert_embed', mock_bert_embed), \
                  patch('guassian_diffusion.tokenize', mock_tokenize), \
                  patch('guassian_diffusion.check_shape', mock_check_shape):
                
                self.diffusion = GaussianDiffusion(
                    denoise_fn=mock_unet,
                    image_size=self.image_size,
                    num_frames=self.num_frames,
                    channels=self.channels,
                    timesteps=self.timesteps,
                    loss_type='l1' # Can test l2 as well
                )

        # Create dummy data
        # Assumes input images are in [0, 1]
        self.x_start = jnp.ones((
            self.batch_size, self.channels, self.num_frames, self.image_size, self.image_size
        ))
        self.t = jnp.array([0, self.timesteps // 2], dtype=jnp.int32) # Example timesteps for batch
        self.noise = jnp.zeros_like(self.x_start)

    def test_initialization(self):
        """Test if the diffusion model initializes correctly."""
        self.assertEqual(self.diffusion.num_timesteps, self.timesteps)
        self.assertEqual(self.diffusion.image_size, self.image_size)
        self.assertEqual(self.diffusion.num_frames, self.num_frames)
        self.assertIsInstance(self.diffusion.alphas_cumprod, nnx.Variable)
        # Check shapes of precomputed variables
        expected_shape = (self.timesteps,)
        self.assertEqual(self.diffusion.alphas_cumprod.value.shape, expected_shape)
        self.assertEqual(self.diffusion.sqrt_alphas_cumprod.value.shape, expected_shape)
        self.assertEqual(self.diffusion.sqrt_one_minus_alphas_cumprod.value.shape, expected_shape)
        self.assertEqual(self.diffusion.posterior_variance.value.shape, expected_shape)

    def test_q_mean_variance(self):
        """Test calculation of q(x_t|x_0) mean and variance."""
        mean, variance, log_variance = self.diffusion.q_mean_variance(self.x_start, self.t)
        self.assertEqual(mean.shape, self.x_start.shape)
        self.assertEqual(variance.shape, (self.batch_size, 1, 1, 1, 1))
        self.assertEqual(log_variance.shape, (self.batch_size, 1, 1, 1, 1))

        # Test at t=0: Compare against the exact formula output
        t0 = jnp.zeros((self.batch_size,), dtype=jnp.int32)
        mean0_actual, var0_actual, _ = self.diffusion.q_mean_variance(self.x_start, t0)

        # Calculate expected values for t=0
        sqrt_alpha_prod_t0 = self.diffusion.sqrt_alphas_cumprod.value[0]
        alpha_prod_t0 = self.diffusion.alphas_cumprod.value[0]
        expected_mean0 = sqrt_alpha_prod_t0 * self.x_start
        expected_var0 = 1.0 - alpha_prod_t0
        # Reshape expected variance for broadcasting comparison
        expected_var0_reshaped = jnp.full((self.batch_size, 1, 1, 1, 1), expected_var0)

        # Assert against calculated expected values
        np.testing.assert_allclose(mean0_actual, expected_mean0, atol=1e-6)
        np.testing.assert_allclose(var0_actual, expected_var0_reshaped, atol=1e-6)

    def test_predict_start_from_noise(self):
        """Test prediction of x_0 from x_t and noise."""
        # Use t > 0 for a meaningful prediction
        t_mid = jnp.full((self.batch_size,), self.timesteps // 2, dtype=jnp.int32)
        # Simulate a noisy image x_t (can use q_sample for this)
        x_t = self.diffusion.q_sample(self.x_start, t=t_mid, noise=self.noise) # Using zero noise here
        
        x_0_pred = self.diffusion.predict_start_from_noise(x_t, t_mid, self.noise)
        self.assertEqual(x_0_pred.shape, self.x_start.shape)
        # If noise is zero, x_0_pred should be close to x_start / sqrt(alpha_cumprod_t) * x_t relationship check
        # For zero noise: x_t = sqrt(alpha_cumprod_t)*x_start
        # predict_start: sqrt(1/alpha_cumprod_t)*x_t - 0 = sqrt(1/alpha_cumprod_t)*sqrt(alpha_cumprod_t)*x_start = x_start
        np.testing.assert_allclose(x_0_pred, self.x_start, atol=1e-4) 

    def test_q_posterior(self):
        """Test calculation of q(x_{t-1}|x_t, x_0) mean and variance."""
        t_mid = jnp.full((self.batch_size,), self.timesteps // 2, dtype=jnp.int32)
        x_t = self.diffusion.q_sample(self.x_start, t=t_mid, noise=self.noise) # Zero noise
        
        mean, var, log_var = self.diffusion.q_posterior(self.x_start, x_t, t_mid)
        self.assertEqual(mean.shape, self.x_start.shape)
        self.assertEqual(var.shape, (self.batch_size, 1, 1, 1, 1))
        self.assertEqual(log_var.shape, (self.batch_size, 1, 1, 1, 1))

    def test_q_sample(self):
        """Test sampling from q(x_t|x_0)."""
        # Test with specific noise
        noise = jax.random.normal(jax.random.PRNGKey(42), self.x_start.shape)
        x_t = self.diffusion.q_sample(self.x_start, self.t, noise=noise)
        self.assertEqual(x_t.shape, self.x_start.shape)

        # Test without providing noise (uses internal randomness - shape check only)
        x_t_rand = self.diffusion.q_sample(self.x_start, self.t, noise=None)
        self.assertEqual(x_t_rand.shape, self.x_start.shape)
        
        # Test at t=0
        # Define the specific noise used for this calculation
        noise_t0 = jax.random.normal(jax.random.PRNGKey(42), self.x_start.shape)
        t0 = jnp.zeros((self.batch_size,), dtype=jnp.int32)
        x_t0 = self.diffusion.q_sample(self.x_start, t0, noise=noise_t0)

        # Calculate the expected output based on the q_sample formula at t=0
        sqrt_alpha_prod_t0 = self.diffusion.sqrt_alphas_cumprod.value[0]
        sqrt_one_minus_alpha_prod_t0 = self.diffusion.sqrt_one_minus_alphas_cumprod.value[0]
        expected_x_t0 = sqrt_alpha_prod_t0 * self.x_start + sqrt_one_minus_alpha_prod_t0 * noise_t0

        # Assert that the actual output matches the formula's expected output
        np.testing.assert_allclose(x_t0, expected_x_t0, atol=1e-6)

    def test_p_mean_variance(self):
        """Test calculation of p(x_{t-1}|x_t) mean and variance (using mock denoise_fn)."""
        x_t = self.diffusion.q_sample(self.x_start, self.t, noise=self.noise) # Zero noise
        
        # Test without clipping
        mean, var, log_var = self.diffusion.p_mean_variance(x_t, self.t, clip_denoised=False)
        self.assertEqual(mean.shape, self.x_start.shape)
        self.assertEqual(var.shape, (self.batch_size, 1, 1, 1, 1))
        self.assertEqual(log_var.shape, (self.batch_size, 1, 1, 1, 1))

        # Test with clipping (mock returns zero noise, so x_recon should be near x_start)
        # Result should be similar to q_posterior with x_start
        mean_clip, _, _ = self.diffusion.p_mean_variance(x_t, self.t, clip_denoised=True)
        self.assertEqual(mean_clip.shape, self.x_start.shape)

    def test_p_sample(self):
        """Test sampling step p(x_{t-1}|x_t)."""
        x_t = jnp.zeros_like(self.x_start)
        t_mid = jnp.full((self.batch_size,), self.timesteps // 2, dtype=jnp.int32)
        
        # Shape test (relies on internal randomness)
        x_prev = self.diffusion.p_sample(x_t, t_mid)
        self.assertEqual(x_prev.shape, x_t.shape)

        # Test at t=0 (nonzero_mask should make noise term zero)
        t0 = jnp.zeros((self.batch_size,), dtype=jnp.int32)
        mean0, _, _ = self.diffusion.p_mean_variance(x_t, t0, clip_denoised=False)
        x_prev_t0 = self.diffusion.p_sample(x_t, t0)
        # Should just return the calculated mean for t=0
        np.testing.assert_allclose(x_prev_t0, mean0, atol=1e-5)

    def test_p_losses(self):
        """Test loss calculation."""
        # Test L1 loss
        loss_l1 = self.diffusion.p_losses(self.x_start, self.t, noise=self.noise) # Target noise is zero
        # Predicted noise is zero (mock), target noise is zero, loss should be zero
        self.assertIsInstance(loss_l1, jax.Array)
        self.assertAlmostEqual(loss_l1.item(), 0.0, places=6)
        
        # Test with non-zero target noise
        target_noise = jnp.ones_like(self.x_start) * 0.5
        loss_l1_nz = self.diffusion.p_losses(self.x_start, self.t, noise=target_noise)
        # Predicted noise is zero, target is 0.5, L1 loss should be 0.5
        self.assertAlmostEqual(loss_l1_nz.item(), 0.5, places=6)
        
        # Test L2 loss
        self.diffusion.loss_type = 'l2'
        loss_l2_nz = self.diffusion.p_losses(self.x_start, self.t, noise=target_noise)
        # Predicted noise is zero, target is 0.5, L2 loss should be mean(0.5^2) = 0.25
        self.assertAlmostEqual(loss_l2_nz.item(), 0.25, places=6)
        self.diffusion.loss_type = 'l1' # Reset for other tests

    def test_call(self):
        """Test the __call__ method (forward pass for training)."""
        # Uses internal random t sampling - primarily check shape and type
        # Patch normalize_img and p_losses to check they are called? More involved.
        loss = self.diffusion(self.x_start)
        self.assertIsInstance(loss, jax.Array)
        self.assertEqual(loss.shape, ()) # Scalar loss expected

    # Tests for p_sample_loop, sample, interpolate are harder to unit test thoroughly
    # They involve loops and significant randomness or calls to other complex methods.
    # Basic shape tests are possible.

    def test_p_sample_loop_shape(self):
        """Test the output shape of the sampling loop."""
        shape = (1, self.channels, self.num_frames, self.image_size, self.image_size)
        # Patching tqdm to avoid printing
        with patch('guassian_diffusion.tqdm', lambda x, **kwargs: x):
             sample = self.diffusion.p_sample_loop(shape)
        self.assertEqual(sample.shape, shape)

    def test_sample_shape(self):
        """Test the main sample method output shape."""
        # Patching tqdm and text utils
        with patch('guassian_diffusion.tqdm', lambda x, **kwargs: x), \
             patch('guassian_diffusion.bert_embed', mock_bert_embed), \
             patch('guassian_diffusion.tokenize', mock_tokenize):
            
            # Test without condition
            samples_no_cond = self.diffusion.sample(batch_size=self.batch_size)
            expected_shape = (self.batch_size, self.channels, self.num_frames, self.image_size, self.image_size)
            self.assertEqual(samples_no_cond.shape, expected_shape)

            # Test with mock condition (infers batch size = 1 from mock_bert_embed)
            samples_with_cond = self.diffusion.sample(cond=["test condition"])
            expected_shape_cond = (1, self.channels, self.num_frames, self.image_size, self.image_size)
            self.assertEqual(samples_with_cond.shape, expected_shape_cond)

    def test_interpolate_shape(self):
        """Test the output shape of interpolation."""
        x1 = jnp.zeros_like(self.x_start)
        x2 = jnp.ones_like(self.x_start)
        with patch('guassian_diffusion.tqdm', lambda x, **kwargs: x):
            interpolated = self.diffusion.interpolate(x1, x2, lam=0.5)
        self.assertEqual(interpolated.shape, self.x_start.shape)

if __name__ == '__main__':
    unittest.main() 