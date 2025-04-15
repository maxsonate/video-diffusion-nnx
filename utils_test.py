# test_utils.py
import unittest
import jax
import jax.numpy as jnp
import numpy as np
# Import all functions being tested
from utils import (
    exists,
    noop,
    is_odd,
    default,
    prob_mask_like,
    clip_grad_norm,
    extract,
    cosine_beta_schedule,
    unnormalize_img,
    normalize_img,
    is_list_str,
    num_to_groups,
    cast_num_frames,
    get_text_from_path,
    identity
)

# Note: Tests for NNX modules (Upsample, Downsample), file I/O (gif/video functions),
# infinite generators (cycle), and PIL interactions (seek_all_images) are omitted
# as they require more complex setup (mocks, test files, NNX context).

class TestUtils(unittest.TestCase):

    # --- Tests for original functions --- 

    def test_exists(self):
        self.assertTrue(exists(1))
        self.assertTrue(exists(0))
        self.assertTrue(exists(""))
        self.assertTrue(exists([]))
        self.assertFalse(exists(None))

    def test_noop(self):
        try:
            noop()
            noop(1, 2, a=3)
        except Exception as e:
            self.fail(f"noop raised {e}")

    def test_is_odd(self):
        self.assertTrue(is_odd(1))
        self.assertTrue(is_odd(3))
        self.assertTrue(is_odd(-1))
        self.assertFalse(is_odd(0))
        self.assertFalse(is_odd(2))
        self.assertFalse(is_odd(-2))

    def test_default(self):
        self.assertEqual(default(5, 10), 5)
        self.assertEqual(default(None, 10), 10)
        self.assertEqual(default(None, lambda: 15), 15)
        self.assertEqual(default("hello", "world"), "hello")
        self.assertEqual(default(0, 10), 0) # Check falsy value

    def test_prob_mask_like(self):
        shape = (10, 10)
        # Note: Cannot test exact probability easily due to internal random key
        mask_all_true = prob_mask_like(shape, 1.0)
        self.assertTrue(jnp.all(mask_all_true))
        self.assertEqual(mask_all_true.shape, shape)
        self.assertEqual(mask_all_true.dtype, jnp.bool_)

        mask_all_false = prob_mask_like(shape, 0.0)
        self.assertTrue(jnp.all(~mask_all_false))
        self.assertEqual(mask_all_false.shape, shape)
        self.assertEqual(mask_all_false.dtype, jnp.bool_)

        mask_half = prob_mask_like(shape, 0.5) 
        self.assertEqual(mask_half.shape, shape)
        self.assertEqual(mask_half.dtype, jnp.bool_)

    def test_clip_grad_norm(self):
        grads_test = {
            'layer1': {'weights': jnp.array([1., 2., 3.]), 'bias': jnp.array([4., 5.])},
            'layer2': {'weights': jnp.array([-2., 1.])}
        }
        max_norm = 1.0
        clipped_grads = clip_grad_norm(grads_test, max_norm, epsilon=1e-9) # Use small epsilon

        self.assertEqual(jax.tree.structure(grads_test), jax.tree.structure(clipped_grads))

        clipped_grad_squared = jax.tree.map(lambda x: jnp.sum(x**2), clipped_grads)
        clipped_l2_norm = jnp.sqrt(jax.tree.reduce(jnp.add, clipped_grad_squared) + 1e-9)
        clipped_total_leaves = len(jax.tree.leaves(clipped_grad_squared))
        clipped_l2_norm_avg = clipped_l2_norm / clipped_total_leaves
        
        self.assertLessEqual(clipped_l2_norm_avg, max_norm + 1e-5) 

    def test_extract(self):
        a = jnp.arange(10) 
        t = jnp.array([1, 3, 5])
        x_shape_4d = (3, 10, 10, 10) # Example 4D shape 
        expected_reshaped_4d = jnp.array([[[[1]]], [[[3]]], [[[5]]]]) # Shape (3, 1, 1, 1)
        
        out = extract(a, t, x_shape_4d)
        np.testing.assert_array_equal(out, expected_reshaped_4d)
        self.assertEqual(out.shape, (t.shape[0],) + (1,) * (len(x_shape_4d) - 1))

    def test_cosine_beta_schedule(self):
        timesteps = 100
        betas = cosine_beta_schedule(timesteps)
        self.assertEqual(betas.shape, (timesteps,))
        self.assertTrue(jnp.all(betas >= 0))
        self.assertTrue(jnp.all(betas <= 1))

    # --- Tests for newly added functions --- 

    def test_unnormalize_img(self):
        normalized = jnp.array([-1., 0., 1.])
        expected = jnp.array([0., 0.5, 1.])
        unnormalized = unnormalize_img(normalized)
        np.testing.assert_allclose(unnormalized, expected, atol=1e-6)

    def test_normalize_img(self):
        unnormalized = jnp.array([0., 0.5, 1.])
        expected = jnp.array([-1., 0., 1.])
        normalized = normalize_img(unnormalized)
        np.testing.assert_allclose(normalized, expected, atol=1e-6)

    def test_is_list_str(self):
        self.assertTrue(is_list_str(["a", "b"]))
        self.assertTrue(is_list_str(("a", "b")))
        self.assertFalse(is_list_str(["a", 1]))
        self.assertFalse(is_list_str("a"))
        # Original logic: all([]) is True, so empty list/tuple returns True
        self.assertTrue(is_list_str([]))
        self.assertTrue(is_list_str(()))
        self.assertFalse(is_list_str([1, 2]))
        self.assertFalse(is_list_str(None))
        self.assertFalse(is_list_str({}))

    def test_num_to_groups(self):
        self.assertEqual(num_to_groups(10, 3), [3, 3, 3, 1])
        self.assertEqual(num_to_groups(9, 3), [3, 3, 3])
        self.assertEqual(num_to_groups(5, 5), [5])
        self.assertEqual(num_to_groups(2, 3), [2])
        self.assertEqual(num_to_groups(0, 3), [])

    def test_cast_num_frames(self):
        # Shape: (channels, frames, height, width)
        t = jnp.ones((3, 10, 4, 4))

        t_equal = cast_num_frames(t, frames=10)
        self.assertEqual(t_equal.shape, (3, 10, 4, 4))
        np.testing.assert_array_equal(t_equal, t)

        t_truncate = cast_num_frames(t, frames=5)
        self.assertEqual(t_truncate.shape, (3, 5, 4, 4))
        np.testing.assert_array_equal(t_truncate, t[:, :5, ...])

        t_pad = cast_num_frames(t, frames=15)
        self.assertEqual(t_pad.shape, (3, 15, 4, 4))
        np.testing.assert_array_equal(t_pad[:, :10, ...], t)
        np.testing.assert_array_equal(t_pad[:, 10:, ...], jnp.zeros((3, 5, 4, 4)))

    def test_get_text_from_path(self):
        path = "/a/b/c/cool-video_test.gif"
        expected = "cool video test"
        self.assertEqual(get_text_from_path(path), expected)
        
        path_simple = "simple.mp4"
        expected_simple = "simple"
        self.assertEqual(get_text_from_path(path_simple), expected_simple)

    def test_identity(self):
        a = object()
        b = [1, 2]
        self.assertIs(identity(a), a)
        self.assertIs(identity(b), b)
        self.assertEqual(identity(5), 5)
        self.assertEqual(identity(5, 1, 2, key=3), 5)


if __name__ == '__main__':
    unittest.main()