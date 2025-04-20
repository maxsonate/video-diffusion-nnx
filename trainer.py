import numpy as np
import jax
import jax.numpy as jnp
import logging
import optax
from flax import nnx
from pathlib import Path
import torch.utils.data as data
from itertools import cycle
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter


# Assuming utils.py contains these functions
from utils import (
    video_array_to_gif, # Only needed if sampling is enabled
    num_to_groups, # Only needed if sampling is enabled
    noop,
    save_checkpoint,
    load_checkpoint,
    clip_grad_norm,
    # cycle # Imported from itertools
)
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, args as ocp_args
from typing import Optional
from datasets import MovingMNIST

# TODO: Add EMA import and functionality
# from ema_pytorch import EMA # Example


class Trainer:
    """Manages the training process for a video diffusion model using Flax NNX.

    Handles dataset loading, optimization, EMA updates (TBD), sampling (TBD), and checkpointing.

    Args:
        diffusion_model: The Flax NNX diffusion model instance to train.
        folder (str): Base directory for results/checkpoints (if specific paths aren't given).
        dataset_path (str): Direct path to the .npy dataset file.
        num_frames (int, optional): Target number of frames per sequence for dataset loading. Defaults to 16.
        train_batch_size (int, optional): Batch size for training. Defaults to 4.
        train_lr (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-4.
        train_num_steps (int, optional): Total number of training steps. Defaults to 100000.
        gradient_accumulate_every (int, optional): Accumulate gradients over N steps. Defaults to 2. (Note: Currently not implemented).
        step_start_ema (int, optional): Step at which to start EMA updates. Defaults to 2000. (Note: EMA not implemented).
        update_ema_every (int, optional): Frequency (in steps) to update EMA weights. Defaults to 10. (Note: EMA not implemented).
        save_and_sample_every (int, optional): Frequency (in steps) to save generated samples. Defaults to 100000. (Note: Sampling logic removed).
        results_folder (str, optional): Directory to save generated samples. Defaults to './results'.
        num_sample_rows (int, optional): Number of rows for the grid in generated sample GIFs. Defaults to 4. (Note: Sampling logic removed).
        max_grad_norm (float | None, optional): Maximum gradient norm for clipping. If None, no clipping is applied. Defaults to None.
        use_path_as_cond (bool, optional): Whether the dataset provides conditioning based on file paths. Defaults to False.
        sample_text (str | None, optional): Text conditioning for sampling (if model supports it). Defaults to None. (Note: Sampling logic removed).
        cond_scale (float, optional): Scale factor for text conditioning during sampling. Defaults to 2.0. (Note: Sampling logic removed).
        checkpoint_every_steps (int, optional): Frequency (in steps) for saving model checkpoints. Defaults to 10.
        checkpoint_dir_path (str, optional): Directory path to save model checkpoints. If empty, defaults to '{results_folder}/checkpoints'. Defaults to ''.
        add_loss_plot (bool, optional): Whether to display a live loss plot (requires plotly/IPython). Defaults to False. (Note: Plotting logic removed).
        tensorboard_dir (str, optional): Directory to save TensorBoard logs. Defaults to ''.
        resume_training_step (int, optional): Step number to resume training from (requires checkpoint loading). Defaults to 0.
        ema_decay (float, optional): Decay rate for EMA. Defaults to 0.9999.
        max_to_keep (int | None, optional): Maximum number of checkpoints to keep. If None, all checkpoints are kept. Defaults to None.
    """
    def __init__(
        self,
        diffusion_model: nnx.Module,
        folder: str, # Used as base for results/checkpoints if not specified
        *,
        dataset_path: str,
        num_frames: int = 16,
        train_batch_size: int = 4,
        train_lr: float = 1e-4,
        train_num_steps: int = 100000,
        gradient_accumulate_every: int = 2,
        step_start_ema: int = 2000,
        update_ema_every: int = 10,
        save_and_sample_every: int = 100000,
        results_folder: str = './results',
        num_sample_rows: int = 4,
        max_grad_norm: float | None = None,
        use_path_as_cond: bool = False,
        sample_text: str | None = None,
        cond_scale: float = 2.0,
        checkpoint_every_steps: int = 10,
        checkpoint_dir_path: str = '',
        add_loss_plot: bool = False, # Kept for potential future use
        tensorboard_dir: str = '',
        resume_training_step: int = 0,
        ema_decay: float = 0.9999,
        max_to_keep: int | None = None,
    ):
        """Initializes the Trainer instance."""
        super().__init__()

        # EMA Configuration
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.ema_decay = ema_decay
        # Initialize EMA parameters as a copy of the model's initial parameters
        _, init_params = nnx.split(diffusion_model)
        self.ema_params = init_params

        # --- Core Components ---
        self.model = diffusion_model
        self.optimizer = nnx.Optimizer(self.model, optax.adam(train_lr))

        # --- Training Configuration ---
        self.train_num_steps = train_num_steps
        self.batch_size = train_batch_size
        self.max_grad_norm = max_grad_norm
        self.use_path_as_cond = use_path_as_cond
        self.gradient_accumulate_every = gradient_accumulate_every # TODO: Implement gradient accumulation

        # --- Dataset and Dataloader ---
        self.image_size = diffusion_model.image_size
        model_num_frames = diffusion_model.num_frames
        # TODO: Make dataset loading more flexible (handle different types/paths)
        print(f"Loading dataset from: {dataset_path}")
        self.ds = MovingMNIST(
            dataset_path,
            image_size=(self.image_size, self.image_size),
            num_frames=model_num_frames, # Use frames from model if not passed explicitly?
            force_num_frames=True
        )
        print(f'Found {len(self.ds)} sequences in dataset.')
        assert len(self.ds) > 0, 'Dataset is empty. Check path and format.'
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True))

        # --- Results and Checkpointing ---
        # Resolve results_folder to an absolute path
        self.results_folder = Path(results_folder).resolve()
        self.results_folder.mkdir(exist_ok=True, parents=True)
        # Resolve checkpoint_dir_path to ensure it's absolute
        self.checkpoint_dir_path = (Path(checkpoint_dir_path).resolve()
                                  if checkpoint_dir_path
                                  else (self.results_folder / 'checkpoints').resolve())
        self.checkpoint_dir_path.mkdir(exist_ok=True, parents=True)
        self.checkpoint_every_steps = checkpoint_every_steps
        # --- Orbax Checkpoint Manager ---
        options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
        self.ckpt_manager = CheckpointManager(self.checkpoint_dir_path, options=options)
        print(f"Checkpoint manager initialized at {self.checkpoint_dir_path} with max_to_keep={max_to_keep}")

        # --- TensorBoard Setup ---
        self.tensorboard_dir = Path(tensorboard_dir).resolve() if tensorboard_dir else self.results_folder / 'tensorboard'
        self.tensorboard_dir.mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        print(f"TensorBoard logs will be saved to: {self.tensorboard_dir}")

        # --- State ---
        self.step = resume_training_step
        if self.step > 0:
            print(f"Attempting to resume training from step {self.step}")
            # TODO: Implement checkpoint loading using load_checkpoint utility
            try:
                self.model = load_checkpoint(self.model, self.step, self.checkpoint_dir_path, ckpt_manager=self.ckpt_manager)
                # TODO: Load optimizer state as well
                print(f"Successfully loaded checkpoint from step {self.step}")
            except FileNotFoundError:
                print(f"Warning: Checkpoint for step {self.step} not found at {self.checkpoint_dir_path}. Starting from step 0.")
                self.step = 0

        # --- Visualization (Not Implemented) ---
        self.add_loss_plot = add_loss_plot

    def sample_batch(self, batch_size):
        """(Placeholder) Sample a single batch of videos.

        Note: Sampling logic and EMA are not currently implemented in the train loop.

        Args:
            batch_size (int): Number of videos to generate in this batch

        Returns:
            Dummy JAX array matching expected sample dimensions.
        """
        print("Warning: sample_batch called, but sampling/EMA is not implemented.")
        # Replace with actual model sampling if/when re-enabled
        shape = (batch_size, self.model.channels, self.model.num_frames, self.image_size, self.image_size)
        return jnp.zeros(shape)

    def train(self, prob_focus_present: float = 0., focus_present_mask = None, log_fn = noop):
        """Runs the main training loop.

        Args:
            prob_focus_present (float, optional): Probability for using guided sampling.
                                                 Defaults to 0.
            focus_present_mask (optional): Mask for guided sampling. Defaults to None.
            log_fn (callable, optional): A function for logging training progress.
                                         Defaults to a no-op.
        """
        assert callable(log_fn)
        print(f"Starting training loop from step {self.step}...")

        # --- Loss Function Definition ---
        # Defined within train to capture self.use_path_as_cond easily
        def compute_loss(model, batch_data, prob_focus_present, focus_present_mask):
            if self.use_path_as_cond:
                # Assuming batch_data is a tuple (video, condition)
                video_data, cond_data = batch_data
                loss = model(
                    video_data,
                    cond=cond_data,
                    prob_focus_present=prob_focus_present,
                    focus_present_mask=focus_present_mask
                )
            else:
                # Assuming batch_data is just the video tensor
                loss = model(
                    batch_data,
                    prob_focus_present=prob_focus_present,
                    focus_present_mask=focus_present_mask
                )
            return loss

        grad_fn = nnx.value_and_grad(compute_loss)

        # --- Training Loop ---
        losses = []
        while self.step < self.train_num_steps:
            # --- Data Loading and Preprocessing ---
            # TODO: Make data loading/preprocessing more robust and configurable
            try:
                batch_torch = next(self.dl)
                batch_data = jnp.array(batch_torch.detach().cpu().numpy())
            except StopIteration:
                print("Dataloader exhausted. Re-initializing.") # Should not happen with cycle
                self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True))
                batch_torch = next(self.dl)
                batch_data = jnp.array(batch_torch.detach().cpu().numpy())

            # Simple normalization (adjust based on actual dataset range)
            if jnp.issubdtype(batch_data.dtype, jnp.integer):
                batch_data = batch_data.astype(jnp.float32) / 255.0
            elif jnp.max(batch_data) > 1.0:
                 # logging.warning(f"Input data max is {jnp.max(batch_data)}. Performing simple max normalization.")
                 batch_data = batch_data / jnp.max(batch_data)

            # --- Gradient Calculation and Optimization ---
            # TODO: Implement gradient accumulation
            # TODO: Implement AMP if feasible for JAX/Flax
            loss, grads = grad_fn(
                self.model,
                batch_data,
                prob_focus_present,
                focus_present_mask,
            )

            # Gradient Clipping
            if self.max_grad_norm is not None:
                grads = clip_grad_norm(grads, max_grad_norm=self.max_grad_norm)

            self.optimizer.update(grads)

            # --- Logging ---
            current_loss = float(np.array(loss))
            losses.append(current_loss)
            print(f"Step: {self.step}/{self.train_num_steps} | Loss: {current_loss:.4f}", flush=True)
            log_fn({'loss': current_loss, 'step': self.step})

            # --- TensorBoard Logging ---
            self.writer.add_scalar('loss/train', current_loss, self.step)

            

            # --- EMA Update ---
            if self.step >= self.step_start_ema and self.step % self.update_ema_every == 0:
                # Extract current model parameters
                _, curr_params = nnx.split(self.model)
                # Update EMA parameters
                self.ema_params = jax.tree_map(
                    lambda ema_p, p: self.ema_decay * ema_p + (1 - self.ema_decay) * p,
                    self.ema_params, curr_params
                )
                print(f"Step: {self.step} | Updated EMA parameters")

            # --- Checkpointing ---
            if self.step > 0 and self.step % self.checkpoint_every_steps == 0:
                print(f"Step: {self.step} | Saving checkpoint...", flush=True)
                try:
                    save_checkpoint(self.ckpt_manager, self.model, self.step)
                    # TODO: Save optimizer state as well
                except Exception as e:
                    print(f"Error saving checkpoint at step {self.step}: {e}")

            # --- Sampling (Removed) ---
            # if self.step != 0 and self.step % self.save_and_sample_every == 0:
            #     milestone = self.step // self.save_and_sample_every
            #     print(f"Step: {self.step} | Generating sample batch at milestone {milestone}...")
            #     self.generate_samples(milestone) # Requires generate_samples method

            self.step += 1
        # --- End of Training Loop ---

        print('Training completed!')
        # Save final checkpoint
        print("Saving final checkpoint...", flush=True)
        try:
            save_checkpoint(self.model, self.step, self.checkpoint_dir_path)
            # TODO: Save final optimizer state
        except Exception as e:
            print(f"Error saving final checkpoint at step {self.step}: {e}")

        # Close TensorBoard writer
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.tensorboard_dir}")
        print(f"View TensorBoard with: tensorboard --logdir={self.tensorboard_dir}")

    # --- Sampling Method (Removed - Placeholder left above) ---
    # def generate_samples(self, milestone: int):
    #     """Generates a batch of samples and saves them as a GIF."""
    #     # ... (Implementation using sample_batch, rearrange, video_array_to_gif) ...
    #     pass