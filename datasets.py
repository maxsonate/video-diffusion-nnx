import numpy as np
from einops import rearrange
from functools import partial

import torch.utils.data as data
import torchvision.transforms as T

# Assuming utils.py contains cast_num_frames and identity
from utils import cast_num_frames, identity

class MovingMNIST(data.Dataset):
    """PyTorch Dataset for Moving MNIST sequences loaded from a .npy file.

    Args:
        file_path (str): Path to the .npy file containing the Moving MNIST data.
                         Expected shape: (num_frames, num_sequences, height, width)
        image_size (int or tuple): The target size (height, width) to resize frames to.
        channels (int, optional): Number of image channels. Defaults to 1.
        num_frames (int, optional): The desired number of frames per sequence.
                                  Defaults to 20.
        horizontal_flip (bool, optional): Whether to apply random horizontal flipping.
                                       Defaults to False.
        force_num_frames (bool, optional): If True, sequences will be padded or truncated
                                         to match num_frames. Defaults to True.
    """
    def __init__(
        self,
        file_path,  # Path to the .npy file
        image_size,  # Target image size (height, width)
        channels=1,  # Number of image channels
        num_frames=20,  # Target number of frames per sequence
        horizontal_flip=False,  # Whether to apply random horizontal flip
        force_num_frames=True,  # Ensure sequences have exactly num_frames
    ):
        super().__init__()
        self.file_path = file_path
        self.image_size = image_size
        self.channnels = channels  # Note: original variable name kept

        self.arrays = np.load(file_path)
        # Rearrange from (f, b, h, w) to (b, f, h, w)
        self.arrays = rearrange(self.arrays, 'f b h w -> b f h w')
        # Add channel dimension: (b, c, f, h, w)
        self.arrays = self.arrays[:, None, ...]  # B C F H W
        self.arrays = self.arrays.astype(np.float32)

        # Partial function to cast sequences to the desired number of frames
        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        """Return the number of sequences in the dataset."""
        return self.arrays.shape[0]

    def __getitem__(self, index):
        """Return a single sequence from the dataset."""
        array = self.arrays[index]
        return self.cast_num_frames_fn(array)