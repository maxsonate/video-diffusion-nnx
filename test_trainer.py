import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from pathlib import Path
import torch
import shutil
import os

# Assuming trainer.py, datasets.py, utils.py are importable
from trainer import Trainer
from datasets import MovingMNIST # Needed for type hints maybe, but dataset itself will be mocked
# Mock utility functions that interact with filesystem or external libs heavily
import utils

# --- Mock Objects ---

class MockDiffusionModel(nnx.Module):
    """A mock NNX Module for the diffusion model."""
    def __init__(self, image_size=32, channels=1, num_frames=10):
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        # Mock parameters needed for optimizer init
        self.mock_param = nnx.Param(jnp.zeros(1))

    def __call__(self, *args, **kwargs):
        # Return a dummy loss (JAX scalar)
        return jnp.array(1.0)

    def sample(self, batch_size, *args, **kwargs):
        # Return dummy samples with expected shape (B, C, F, H, W)
        shape = (batch_size, self.channels, self.num_frames, self.image_size, self.image_size)
        return jnp.zeros(shape)

# --- Test Class ---

class TestTrainer(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("./test_trainer_temp")
        self.test_dir.mkdir(exist_ok=True)
        self.results_folder = self.test_dir / "results"
        self.checkpoint_dir = self.test_dir / "checkpoints"
        # Explicitly create the checkpoint directory for the test
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.dummy_dataset_path = str(self.test_dir / "dummy_data.npy")

        # Create a small dummy npy file
        dummy_data = np.zeros((5, 2, 32, 32), dtype=np.uint8) # F, B, H, W
        np.save(self.dummy_dataset_path, dummy_data)

        # Instantiate the mock model
        self.mock_model = MockDiffusionModel(image_size=64, channels=1, num_frames=10)

        # Basic Trainer args
        self.trainer_args = {
            "diffusion_model": self.mock_model,
            "folder": str(self.test_dir),
            "dataset_path": self.dummy_dataset_path,
            "train_batch_size": 2,
            "train_lr": 1e-4,
            "train_num_steps": 5,
            "results_folder": str(self.results_folder),
            "checkpoint_dir_path": str(self.checkpoint_dir),
            "save_and_sample_every": 3,
            "checkpoint_every_steps": 2,
        }

        # --- Patch external dependencies ---
        # Patch dataset loading (assuming MovingMNIST is used internally)
        # We need to patch the *instance* used by the Trainer
        # Remove autospec=True to avoid potential spec creation issues
        self.patcher_dataset = patch('trainer.MovingMNIST')
        self.MockMovingMNIST = self.patcher_dataset.start()
        self.mock_dataset_instance = self.MockMovingMNIST.return_value
        self.mock_dataset_instance.__len__.return_value = 2 # Needs > 0 sequences
        # Make the dataloader yield mock torch tensors
        self.mock_dataloader = MagicMock()
        dummy_batch = torch.zeros((2, 1, 10, 64, 64)) # B, C, F, H, W
        self.mock_dataloader.__next__.return_value = dummy_batch
        self.patcher_dataloader = patch('trainer.cycle', return_value=self.mock_dataloader)
        self.patcher_dataloader.start()

        # Patch utils functions
        self.patcher_save_checkpoint = patch('trainer.save_checkpoint')
        self.mock_save_checkpoint = self.patcher_save_checkpoint.start()

        self.patcher_clip_grad = patch('trainer.clip_grad_norm', side_effect=lambda grads, *, max_grad_norm: grads)
        self.mock_clip_grad = self.patcher_clip_grad.start()

        self.patcher_vid_gif = patch('trainer.video_array_to_gif')
        self.mock_vid_gif = self.patcher_vid_gif.start()

        self.patcher_num_groups = patch('trainer.num_to_groups', return_value=[1, 1])
        self.mock_num_groups = self.patcher_num_groups.start()


    def tearDown(self):
        """Clean up test environment."""
        # Stop all patchers
        self.patcher_dataset.stop()
        self.patcher_dataloader.stop()
        self.patcher_save_checkpoint.stop()
        self.patcher_clip_grad.stop()
        self.patcher_vid_gif.stop()
        self.patcher_num_groups.stop()

        # Remove temporary directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test if the Trainer initializes correctly."""
        trainer = Trainer(**self.trainer_args)
        self.assertEqual(trainer.batch_size, 2)
        self.assertEqual(trainer.train_num_steps, 5)
        self.assertEqual(trainer.step, 0)
        self.assertTrue(self.results_folder.exists())
        self.assertTrue(self.checkpoint_dir.exists())
        self.assertIsInstance(trainer.optimizer, nnx.Optimizer)
        # Check if the hardcoded MovingMNIST was attempted to be initialized
        # Note: This depends on the hardcoded path inside Trainer.__init__
        # If the path changes, this assertion needs update or removal.
        self.MockMovingMNIST.assert_called_once_with(
             # '/content/gdrive/MyDrive/Research/Data/mnist_test_seq_100.npy', # The hardcoded path
             self.dummy_dataset_path, # Check for the dummy path now
             image_size=(64, 64), # Default image size
             num_frames=10, # From mock model
             force_num_frames=True
        )


    def test_train_loop_runs_basic(self):
        """Test if the train loop runs for a few steps without crashing."""
        trainer = Trainer(**self.trainer_args)
        try:
            trainer.train()
        except Exception as e:
            self.fail(f"trainer.train() raised exception unexpectedly: {e}")
        self.assertEqual(trainer.step, self.trainer_args['train_num_steps'])

    def test_checkpointing_called(self):
        """Test if save_checkpoint is called at the correct interval."""
        args = self.trainer_args.copy()
        args['train_num_steps'] = 5
        args['checkpoint_every_steps'] = 2
        trainer = Trainer(**args)
        trainer.train()
        # Expected calls at step 2, 4 (inside loop) and 5 (final)
        self.assertEqual(self.mock_save_checkpoint.call_count, 3)
        expected_calls = [
            call(trainer.model, 2, trainer.checkpoint_dir_path),
            call(trainer.model, 4, trainer.checkpoint_dir_path),
            call(trainer.model, 5, trainer.checkpoint_dir_path) # Check for step 5
        ]
        self.mock_save_checkpoint.assert_has_calls(expected_calls, any_order=False) # Ensure order


    def test_gradient_clipping_called(self):
        """Test if clip_grad_norm is called when max_grad_norm is set."""
        args = self.trainer_args.copy()
        args['max_grad_norm'] = 1.0
        args['train_num_steps'] = 2
        trainer = Trainer(**args)
        trainer.train()
        self.assertGreater(self.mock_clip_grad.call_count, 0)
        # Check if it was called with the correct keyword argument
        self.mock_clip_grad.assert_called_with(unittest.mock.ANY, max_grad_norm=1.0)


    def test_gradient_clipping_not_called(self):
        """Test if clip_grad_norm is NOT called when max_grad_norm is None."""
        args = self.trainer_args.copy()
        args['max_grad_norm'] = None
        args['train_num_steps'] = 2
        trainer = Trainer(**args)
        trainer.train()
        self.mock_clip_grad.assert_not_called()


if __name__ == '__main__':
    unittest.main() 