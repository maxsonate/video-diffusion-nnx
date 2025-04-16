# test_utils.py
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import shutil
from pathlib import Path
import orbax.checkpoint as ocp # Import orbax for potential error checking
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
    identity,
    save_checkpoint,
    load_checkpoint
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


# --- Mock Model for Testing ---

class MockSimpleNNXModel(nnx.Module):
    def __init__(self, value: float, *, rngs: nnx.Rngs):
        self.param = nnx.Param(jnp.array([value]))
        # Add another type of state for more robust testing
        self.counter = nnx.Variable(jnp.array(0))

    def increment(self):
        self.counter.value += 1

# --- Test Class ---

class TestCheckpointUtils(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for checkpoints."""
        self.test_dir = Path("./test_checkpoint_temp_utils") # Unique name
        self.test_dir.mkdir(exist_ok=True)
        # Resolve to an absolute path as required by Orbax
        self.checkpoint_path = str(self.test_dir.resolve())
        self.step = 42

        # Initialize a mock model
        self.key = jax.random.PRNGKey(0)
        self.model_to_save = MockSimpleNNXModel(value=1.23, rngs=nnx.Rngs(0))
        self.model_to_save.increment() # Modify state (counter becomes 1)

    def tearDown(self):
        """Remove the temporary directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_save_checkpoint_creates_files(self):
        """Test that save_checkpoint creates the expected directory structure."""
        save_checkpoint(self.model_to_save, self.step, self.checkpoint_path)

        # Check if the Orbax directory structure exists
        expected_checkpoint_dir = self.test_dir / str(self.step) / "state"
        self.assertTrue(expected_checkpoint_dir.exists(), f"Checkpoint directory {expected_checkpoint_dir} not found.")
        # Check if the directory is not empty
        self.assertTrue(any(expected_checkpoint_dir.iterdir()), f"Checkpoint directory {expected_checkpoint_dir} is empty.")

    def test_load_checkpoint_restores_state(self):
        """Test that load_checkpoint correctly restores the model's state."""
        # 1. Save the initial model
        save_checkpoint(self.model_to_save, self.step, self.checkpoint_path)

        # 2. Create a new model instance with a different initial state
        model_to_load = MockSimpleNNXModel(value=9.99, rngs=nnx.Rngs(1)) # Different value and rngs
        self.assertEqual(model_to_load.counter.value, 0) # Counter starts at 0

        # 3. Load the checkpoint into the new model
        loaded_model = load_checkpoint(model_to_load, self.step, self.checkpoint_path)

        # 4. Verify the state is restored
        self.assertIsInstance(loaded_model, MockSimpleNNXModel)

        # Split both models to compare states
        _, state_original = nnx.split(self.model_to_save)
        _, state_loaded = nnx.split(loaded_model)

        # Compare parameter values (should be ~1.23)
        np.testing.assert_allclose(
            state_loaded['param'].value,
            state_original['param'].value,
            rtol=1e-6,
            err_msg="Loaded parameter value does not match saved value."
        )
        # Compare other variable values (counter should be 1)
        np.testing.assert_array_equal(
            state_loaded['counter'].value,
            state_original['counter'].value,
            err_msg="Loaded variable (counter) value does not match saved value."
        )
        # Also check the counter value directly on the loaded model object
        self.assertEqual(loaded_model.counter.value, 1, "Counter value on loaded model object is incorrect.")

    def test_load_nonexistent_checkpoint_raises_error(self):
        """Test that loading a non-existent step raises an error."""
        model_to_load = MockSimpleNNXModel(value=0.0, rngs=nnx.Rngs(0))
        non_existent_step = 999
        # Orbax checkpointer.restore() should raise an error if the path doesn't exist
        # Catching a more specific error if possible, but Exception is safer across versions
        with self.assertRaises((FileNotFoundError, Exception), msg="Loading non-existent checkpoint did not raise an error."):
            load_checkpoint(model_to_load, non_existent_step, self.checkpoint_path)


if __name__ == '__main__':
    unittest.main()