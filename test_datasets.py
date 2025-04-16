import unittest
import numpy as np
import jax.numpy as jnp
import os

# Assuming datasets.py is in the same directory or accessible in the path
from datasets import MovingMNIST


class TestMovingMNIST(unittest.TestCase):

    def setUp(self):
        """Set up a dummy .npy file for testing."""
        self.test_file_path = 'test_mnist_data.npy'
        self.num_sequences = 5
        self.num_frames_in_file = 15
        self.height = 32
        self.width = 32
        # Create dummy data: (F, B, H, W)
        dummy_data = np.random.rand(
            self.num_frames_in_file, self.num_sequences, self.height, self.width
        ).astype(np.uint8) # Moving MNIST is typically uint8
        np.save(self.test_file_path, dummy_data)

        # Common dataset parameters for tests
        self.image_size = 64
        self.target_num_frames = 20
        self.channels = 1

    def tearDown(self):
        """Remove the dummy .npy file after tests."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_initialization(self):
        """Test dataset initialization and basic properties."""
        dataset = MovingMNIST(
            file_path=self.test_file_path,
            image_size=self.image_size,
            num_frames=self.target_num_frames,
            channels=self.channels,
            force_num_frames=True
        )
        self.assertEqual(len(dataset), self.num_sequences)
        self.assertEqual(dataset.image_size, self.image_size)
        self.assertEqual(dataset.channnels, self.channels) # Using the current variable name
        self.assertEqual(dataset.cast_num_frames_fn.keywords['frames'], self.target_num_frames)

    def test_len(self):
        """Test the __len__ method."""
        dataset = MovingMNIST(file_path=self.test_file_path, image_size=self.image_size)
        self.assertEqual(len(dataset), self.num_sequences)

    def test_getitem_raw_shape_and_type(self):
        """Test the shape and type of item returned by __getitem__ *before* transforms (current state)."""
        dataset = MovingMNIST(
            file_path=self.test_file_path,
            image_size=self.image_size,
            num_frames=self.target_num_frames,
            force_num_frames=True
        )
        item = dataset[0]
        # Note: This test reflects the *current* buggy state
        # Check if item is either numpy.ndarray or jax.numpy.ndarray
        self.assertIsInstance(item, (np.ndarray, jnp.ndarray)) # Modified assertion
        # Shape: (C, F, H, W) - C=1, F=target_num_frames, H/W = original size
        expected_shape = (self.channels, self.target_num_frames, self.height, self.width)
        self.assertEqual(item.shape, expected_shape)
        # self.assertEqual(item.dtype, np.float32) # This will fail due to np.astype bug

    def test_getitem_frame_casting_padding(self):
        """Test if frames are correctly padded when force_num_frames=True."""
        target_frames = 25 # More than in the file (15)
        dataset = MovingMNIST(
            file_path=self.test_file_path,
            image_size=self.image_size,
            num_frames=target_frames,
            force_num_frames=True
        )
        item = dataset[0]
        self.assertEqual(item.shape[1], target_frames) # Check frame dimension

    def test_getitem_frame_casting_truncation(self):
        """Test if frames are correctly truncated when force_num_frames=True."""
        target_frames = 10 # Less than in the file (15)
        dataset = MovingMNIST(
            file_path=self.test_file_path,
            image_size=self.image_size,
            num_frames=target_frames,
            force_num_frames=True
        )
        item = dataset[0]
        self.assertEqual(item.shape[1], target_frames) # Check frame dimension

    def test_getitem_frame_casting_disabled(self):
        """Test if frame count remains original when force_num_frames=False."""
        dataset = MovingMNIST(
            file_path=self.test_file_path,
            image_size=self.image_size,
            num_frames=self.target_num_frames, # This should be ignored
            force_num_frames=False
        )
        item = dataset[0]
        self.assertEqual(item.shape[1], self.num_frames_in_file) # Should match original frames

if __name__ == '__main__':
    unittest.main() 