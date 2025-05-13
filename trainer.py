import os
import time
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import numpy as np
import jax
import jax.numpy as jnp
import logging
import optax
from flax import nnx
from pathlib import Path
import torch.utils.data as data
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from functools import partial
from flax import struct
from jax.profiler import start_trace, stop_trace
import jax.profiler

from jax.experimental.pjit import pjit


# Assuming utils.py contains these functions
from utils import (
    video_array_to_gif, # Only needed if sampling is enabled
    num_to_groups, # Only needed if sampling is enabled
    noop,
    save_checkpoint,
    load_checkpoint,
    clip_grad_norm,
    # clip_grad_norm_with_tb_logging, # Removed
    # cycle # Imported from itertools
)
import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions #, args as ocp_args # Removed
from typing import Optional
from datasets import MovingMNIST

# TODO: Add EMA import and functionality
# from ema_pytorch import EMA # Example

@struct.dataclass
class NnxTrainState:
    # Dynamic parts (potentially sharded)
    params: nnx.State
    opt_state: optax.OptState
    ema_params: nnx.State
    # graphdef and tx removed - they will be passed as static args

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
        lr_decay_start_step (int, optional): Step number to start learning rate decay. Defaults to 0.
        lr_decay_steps (int, optional): Number of steps over which to decay learning rate. Defaults to 0.
        lr_decay_coeff (float, optional): Coefficient for learning rate decay. Defaults to 1.0.
        rng_seed (int, optional): Master PRNG seed for reproducibility. Defaults to 0.
        profile_flush_interval_steps (int, optional): Frequency (in steps) for flushing JAX profiler file traces. Defaults to 1000.
    """
    def __init__(
        self,
        diffusion_model: nnx.Module,
        folder: str, # Used as base for results/checkpoints if not specified
        *,
        rng_seed: int = 0,
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
        lr_decay_start_step: int = 0,
        lr_decay_steps: int = 0,
        lr_decay_coeff: float = 1.0,
        profile_flush_step: int = 100,
        num_model_shards: int = 1
    ):
        """Initializes the Trainer instance."""
        super().__init__()

        # --- PRNG Key Setup ---
        self.key = jax.random.PRNGKey(rng_seed)
        self.profile_flush_step = profile_flush_step # Store the interval

        # EMA Configuration
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.ema_decay = ema_decay

        # --- Core Components ---
        self.model = diffusion_model
        self.graphdef, init_params = nnx.split(self.model) # Split once

        self.lr_schedule = optax.piecewise_interpolate_schedule(
            interpolate_type='cosine',
            init_value=train_lr,
            boundaries_and_scales={
                lr_decay_start_step:                    1.0,             # hold at init_value until you hit decay
                lr_decay_start_step + lr_decay_steps:  lr_decay_coeff  # then cosine‐interpolate down to init*coeff
            }
        )
        # Create the Optax transformer (optimizer logic)
        self.tx = optax.adam(self.lr_schedule)

        # Initialize parameters, optimizer state, and EMA state
        init_opt_state = self.tx.init(init_params)
        init_ema_params = jax.tree_util.tree_map(lambda x: x, init_params)

        # --- Training Configuration ---
        self.train_num_steps = train_num_steps
        self.batch_size = train_batch_size
        self.max_grad_norm = max_grad_norm
        self.use_path_as_cond = use_path_as_cond
        self.gradient_accumulate_every = gradient_accumulate_every # TODO: Implement gradient accumulation

        # --- Device Setup ---
        self.n_devices = jax.local_device_count()
        devices = jax.local_devices()
        assert self.batch_size % self.n_devices == 0, "batch_size must be divisible by number of devices"
        self.per_device_bs = self.batch_size // self.n_devices

        # --- Create and Replicate Train State ---
        # Define model axis for model parallelism
        self.num_model_shards = num_model_shards # Example: 2-stage pipeline.
        assert self.n_devices % self.num_model_shards == 0, "Number of devices must be divisible by num_model_shards"
        data_parallel_size = self.n_devices // self.num_model_shards
        model_axis_name = 'model'
        data_axis_name = 'data'
        self.model_axis_name = model_axis_name # Store on self
        self.data_axis_name = data_axis_name   # Store on self

        device_mesh = mesh_utils.create_device_mesh((data_parallel_size, self.num_model_shards))
        self.mesh = Mesh(devices=device_mesh, axis_names=(self.data_axis_name, self.model_axis_name)) # Use self. attributes
        logging.info(f"Created mesh with shape: {self.mesh.shape} and axis_names: {self.mesh.axis_names}")

        # Parameters for Unet3D constructor (it now expects mesh and model_axis_name)
        # This assumes self.model is an instance of the updated Unet3D
        # If self.model was already instantiated, this change implies it needs re-instantiation
        # or that its existing __init__ is compatible / mesh is set post-init.
        # For NNX, this is fine as the model object definition itself is less tied to sharding
        # until its parameters are extracted and sharded.
        # We ensure the model object has these attributes if it uses them internally.
        if hasattr(self.model, 'mesh') and hasattr(self.model, 'model_axis_name'):
            self.model.mesh = self.mesh
            self.model.model_axis_name = self.model_axis_name # Use self.attribute
        
        # Create parameter sharding rules
        abstract_params = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), init_params)
        
        # Generate PartitionSpec PyTree for params
        # Correctly convert diverse path entries (GetAttrKey, SequenceKey, DictKey) to strings.
        _param_path_to_name_tuple = lambda path: tuple(Trainer._convert_path_entry_to_str(p) for p in path)
        
        state_param_sharding_spec = jax.tree_util.tree_map_with_path(
            lambda path, x: self._get_param_sharding(_param_path_to_name_tuple(path), x),
            abstract_params
        )

        # Sharding for opt_state: should mirror params sharding
        # This assumes opt_state structure is a PyTree mirroring params, or that
        # a similar path-based logic can apply.
        # For Adam, states are often per-parameter.
        abstract_opt_state = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if hasattr(x, 'shape') else x, init_opt_state)
        # Create the optimizer state sharding spec using a helper method
        opt_state_sharding_spec = Trainer._create_optimizer_sharding_spec(
            state_param_sharding_spec,
            abstract_opt_state
        )
        
        # This creates a Pytree of PartitionSpecs for opt_state, mirroring params.
        
        ema_params_sharding_spec = jax.tree_util.tree_map(
            Trainer._get_opt_or_ema_sharding,
            state_param_sharding_spec,
            abstract_params # EMA params mirror original params structure
        )

        host_state = NnxTrainState(
            params=init_params,
            opt_state=init_opt_state,
            ema_params=init_ema_params
            # graphdef and tx are kept as self.graphdef, self.tx
        )

        train_state_sharding_specs_tree = NnxTrainState(
            params=state_param_sharding_spec,
            opt_state=opt_state_sharding_spec,
            ema_params=ema_params_sharding_spec
        )
        
        self.train_state_sharding = jax.tree_util.tree_map(
            lambda spec: NamedSharding(self.mesh, spec),
            train_state_sharding_specs_tree
        )
        
        self.state = host_state # Pass host state to pjit, in_shardings will handle it

        # --- Dataset and Dataloader ---
        self.image_size = diffusion_model.image_size
        model_num_frames = diffusion_model.num_frames
        # TODO: Make dataset loading more flexible (handle different types/paths)
        logging.info(f"Loading dataset from: {dataset_path}")
        self.ds = MovingMNIST(
            dataset_path,
            image_size=(self.image_size, self.image_size),
            num_frames=model_num_frames, # Use frames from model if not passed explicitly?
            force_num_frames=True
        )
        # Get the total number of samples in the dataset
        num_samples = len(self.ds)
        logging.info(f"Found {num_samples} sequences in dataset.")
        assert num_samples > 0, "Dataset is empty. Check path and format."
        # Add drop_last=True to ensure all batches have the expected size
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True))

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
        logging.info(f"Checkpoint manager initialized at {self.checkpoint_dir_path} with max_to_keep={max_to_keep}")

        # --- TensorBoard Setup ---
        self.tensorboard_dir = Path(tensorboard_dir).resolve() if tensorboard_dir else self.results_folder / 'tensorboard'
        self.tensorboard_dir.mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        logging.info(f"TensorBoard logs will be saved to: {self.tensorboard_dir}")

        # --- State ---
        self.step = resume_training_step
        if self.step > 0:
            logging.info(f"Attempting to resume training from step {self.step}")
            # Adapt checkpoint loading for NnxTrainState
            try:
                model, ema_params = load_checkpoint(self.model, self.step, self.checkpoint_dir_path, self.ckpt_manager) 
                if model is not None and ema_params is not None:
                   host_state = NnxTrainState(
                       params=nnx.split(model)[1],
                       opt_state=init_opt_state,
                       ema_params=ema_params
                   ) # Update host_state before replication
                   logging.info(f"Successfully loaded checkpoint state for step {self.step}")
                else:
                   logging.warning(f"Checkpoint loading function returned None for step {self.step}.")
                   self.step = 0
            except FileNotFoundError:
                logging.warning(f"Checkpoint for step {self.step} not found at {self.checkpoint_dir_path}.")
                self.step = 0

            self.state = host_state
        # --- Visualization (Not Implemented) ---
        self.add_loss_plot = add_loss_plot

        # Define pjit'd train step with shardings
        _p_train_step_in_shardings = (
            self.train_state_sharding,  # <--- HERE it is used for the input state
            NamedSharding(self.mesh, P(self.data_axis_name, None)),  # batch_data
            NamedSharding(self.mesh, P()),  # key
            NamedSharding(self.mesh, P()),  # step
            # Static args don't need shardings here
             NamedSharding(self.mesh, P()), # step_start_ema
             NamedSharding(self.mesh, P()), # update_ema_every
             NamedSharding(self.mesh, P())  # ema_decay
        )
        _p_train_step_out_shardings = (
            self.train_state_sharding, # <--- AND HERE for the output state
            NamedSharding(self.mesh, P())   # loss
        )

        @partial(pjit,
                 in_shardings=_p_train_step_in_shardings, # <--- Passed to pjit
                 out_shardings=_p_train_step_out_shardings, # <--- Passed to pjit
                 static_argnums=(4, 5, 6) 
                )
        def _pjit_train_step(state: NnxTrainState, batch_data, key, step: int,
                              graphdef: nnx.GraphDef, 
                              tx: optax.GradientTransformation, 
                              use_path_as_cond: bool, 
                              step_start_ema: int, 
                              update_ema_every: int, 
                              ema_decay: float):
            """Performs a single training step on each device."""

            # 'step' is now a static argument
            def loss_fn(params):
                # Use the passed static graphdef
                model_for_loss = nnx.merge(graphdef, params)

                # Use the passed-in static arg 'use_path_as_cond'
                if use_path_as_cond:
                    video_data, cond_data = batch_data
                    loss = model_for_loss(
                        video_data,
                        key=key,
                        cond=cond_data,
                        prob_focus_present=0., # Pass these if needed
                        focus_present_mask=None
                    )
                else:
                    loss = model_for_loss(
                        batch_data,
                        key=key,
                        prob_focus_present=0.,
                        focus_present_mask=None
                    )
                return loss

            # Calculate loss and gradients using the state's params
            (loss, grads) = jax.value_and_grad(loss_fn)(state.params)

            # Gradients and loss are implicitly averaged/summed by pjit based on operations
            # (e.g., mean reduction in loss_fn)

            # Apply optimizer updates using the passed static tx and state's opt_state
            updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)

            # --- EMA Update ---
            # Access ema_decay etc. from passed-in static args
            # 'step' is static
            do_ema = jnp.logical_and(step >= step_start_ema,
                                     (step % update_ema_every) == 0)
            new_ema_params = jax.lax.cond(
                do_ema,
                # Use the passed-in static arg 'ema_decay'
                lambda _: jax.tree_util.tree_map(lambda ema_p, p: ema_decay * ema_p + (1 - ema_decay) * p,
                                       state.ema_params, new_params),
                lambda _: state.ema_params,
                operand=None
            )

            # Return the updated state (only dynamic parts) and the scalar loss
            new_state = state.replace(
                params=new_params,
                opt_state=new_opt_state,
                ema_params=new_ema_params
            )
            return new_state, loss

        self.p_train_step = _pjit_train_step

    @staticmethod
    def _convert_path_entry_to_str(p):
        if isinstance(p, jax.tree_util.GetAttrKey):
            return p.name
        elif isinstance(p, jax.tree_util.SequenceKey):
            return str(p.idx) # Convert index to string for consistency in the path tuple
        elif isinstance(p, jax.tree_util.DictKey):
            return p.key # DictKey often has a .key attribute that is the actual key
        elif hasattr(p, 'key') and p.key is not None: # Fallback for other key-like objects
            return str(p.key) # Or p.key if it's already a string
        else:
            return str(p) # Generic fallback

    def _get_param_sharding(self, param_path_tuple, param_leaf):
        # Initialize all parameter dimensions to be replicated on all mesh axes
        param_spec_list = [None] * param_leaf.ndim 

        leaf_name = param_path_tuple[-1] if param_path_tuple else ""

        # Model parallelism for specific layers' weights and biases:
        # Shard the last dimension of kernels/weights and biases along the self.model_axis_name.
        # Parameters are replicated by default on other mesh axes (including data_axis_name).
        is_kernel_or_weight = leaf_name in ['kernel', 'w']
        is_bias = leaf_name == 'bias' # For nnx.Conv bias
        if leaf_name == 'b' and any(parent_key in str(param_path_tuple) for parent_key in ['Linear', 'mlp']): # Heuristic for nnx.Linear bias
            is_bias = True

        if param_leaf.ndim > 0 and (is_kernel_or_weight or is_bias):
            param_spec_list[-1] = self.model_axis_name # Target last dim for model sharding
            # If param_leaf.ndim == 1 (e.g. bias), its single dimension is now sharded on self.model_axis_name.
            # All other dimensions (if any) remain None, meaning replicated on other mesh axes (like data_axis_name).

        return P(*param_spec_list) if param_leaf.ndim > 0 else P() # For scalar params (fully replicated)

    @staticmethod
    def _get_opt_or_ema_sharding(param_sharding_spec_leaf, opt_ema_leaf_struct):
        # If opt_ema_leaf is scalar or not a JAX type, replicate.
        if not hasattr(opt_ema_leaf_struct, 'ndim') or opt_ema_leaf_struct.ndim == 0:
            return P()
        # Else, use the sharding spec of the corresponding parameter.
        return param_sharding_spec_leaf

    @staticmethod
    def _create_optimizer_sharding_spec(state_param_sharding_spec, abstract_opt_state):
        """Creates the sharding specification Pytree for the optimizer state.
        
        Handles typical Optax optimizer structures (like AdamState or a tuple containing it)
        where parts of the state (e.g., mu, nu) mirror the parameter structure,
        while others (e.g., count) are scalars or other simple states.
        """
        # Optax optimizer states can be tuples, where the Adam-like state
        # (with mu, nu, count) is often the first element.
        if not isinstance(abstract_opt_state, tuple):
            # If it's not a tuple, assume it's the Adam-like state directly (e.g. ScaleByAdamState)
            adam_like_state_abstract = abstract_opt_state
            other_states_abstract = ()
        else:
            # Assuming the first element is the one with mu, nu, count (e.g., ScaleByAdamState)
            adam_like_state_abstract = abstract_opt_state[0]
            other_states_abstract = abstract_opt_state[1:]

        # mu and nu fields have PyTrees matching params structure, accessed from adam_like_state_abstract
        mu_sharding_spec = jax.tree_util.tree_map(
            Trainer._get_opt_or_ema_sharding, # Use the static helper
            state_param_sharding_spec, # Structure for params
            adam_like_state_abstract.mu # mu part of the identified Adam-like state
        )
        nu_sharding_spec = jax.tree_util.tree_map(
            Trainer._get_opt_or_ema_sharding, # Use the static helper
            state_param_sharding_spec, # Structure for params
            adam_like_state_abstract.nu  # nu part of the identified Adam-like state
        )
        # Optimizer state often has a scalar 'count' component in its Adam-like part.
        count_sharding_spec = P() # Assuming count is always a scalar and replicated

        # Reconstruct the sharding for the Adam-like part of the state
        try:
            adam_like_sharding_spec = type(adam_like_state_abstract)(
                count=count_sharding_spec,
                mu=mu_sharding_spec,
                nu=nu_sharding_spec
            )
        except TypeError as e:
            logging.error(f"Error creating Adam-like part of optimizer sharding spec. Check structure: {e}")
            logging.error(f"Adam-like Abstract State Type: {type(adam_like_state_abstract)}")
            logging.error(f"Adam-like Abstract State: {adam_like_state_abstract}")
            raise ValueError("Could not construct Adam-like sharding spec, incompatible structure.") from e

        # Reconstruct the full optimizer state sharding spec (tuple or single object)
        if not isinstance(abstract_opt_state, tuple):
            optimizer_sharding_spec = adam_like_sharding_spec
        else:
            # Shard other states in the tuple (often EmptyState, which is scalar-like)
            other_sharding_specs = tuple(P() for _ in other_states_abstract) # Assume replicated
            optimizer_sharding_spec = (adam_like_sharding_spec,) + other_sharding_specs
            
        return optimizer_sharding_spec

    def sample_batch(self, batch_size):
        """(Placeholder) Sample a single batch of videos.

        Note: Sampling logic and EMA are not currently implemented in the train loop.

        Args:
            batch_size (int): Number of videos to generate in this batch

        Returns:
            Dummy JAX array matching expected sample dimensions.
        """
        logging.warning("Warning: sample_batch called, but sampling/EMA is not implemented.")
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
        logging.info(f"Starting training loop from step {self.step}...")

        # --- Training Loop ---
        losses = []
        key = self.key
        jax.profiler.start_server(9999) # For live profiling (e.g., Perfetto UI)

        # --- File-based JAX Profiler Setup ---
        if self.profile_flush_step > 0:
            _trace_log_dir = self.tensorboard_dir # Using existing path for file traces
            Path(_trace_log_dir).mkdir(parents=True, exist_ok=True) # Ensure directory exists
            
            # Start file tracing if an interval is set, or if the original logic implies it.
            # The original code started it unconditionally. We will keep that, and control flushing with the interval.
            jax.profiler.start_trace(_trace_log_dir, create_perfetto_link=False)
            logging.info(f"JAX profiler file trace started. Traces will be saved to '{_trace_log_dir}'.")
            if self.profile_flush_step > 0:
                logging.info(f"File traces will be flushed at {self.profile_flush_step} steps.")


        while self.step < self.train_num_steps:
            # --- Split Key for the current step ---
            key, step_key = jax.random.split(key)

            # --- Data Loading and Preprocessing ---
            # TODO: Make data loading/preprocessing more robust and configurable
            try:
                batch_torch = next(self.dl)
                batch_data = jnp.array(batch_torch.detach().cpu().numpy())
            except StopIteration:
                logging.warning("Dataloader exhausted unexpectedly. Re-initializing.")
                self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True))
                batch_torch = next(self.dl)
                batch_data = jnp.array(batch_torch.detach().cpu().numpy())


            # --- Execute Pjit'd Step ---
            # Pass self.state and the required static values explicitly
            # Call the pjit'd function with explicit device shardings (no mesh context required)
            start_time = time.time()
            # No need for mesh context with SingleDeviceSharding
            self.state, loss_val = self.p_train_step(
                self.state,           # Arg 0 (replicated state)
                batch_data,          # Arg 1 (unsharded host data - pjit handles sharding)
                step_key,           # Arg 2 (replicated key)
                # Dynamic args (non-static) start here
                self.step,           # Arg 3 (step) - Scalar passed directly
                self.graphdef,       # Arg 4: graphdef (static)
                self.tx,             # Arg 5 (tx)
                self.use_path_as_cond,# Arg 6 (use_path_as_cond)
                self.step_start_ema, # Arg 7 (step_start_ema) - Scalar
                self.update_ema_every,# Arg 8 (update_ema_every) - Scalar
                self.ema_decay       # Arg 9 (ema_decay) - Scalar
            )

            end_time = time.time()
            
            self.writer.add_scalar('step_time', end_time - start_time, self.step)

            # --- Logging ---
            # loss_val is the averaged scalar loss, potentially still on device
            # Use jax.device_get to bring it to the host CPU before converting
            current_loss = float(jax.device_get(loss_val))
            losses.append(current_loss)
            logging.info(f"Step: {self.step}/{self.train_num_steps} | Loss: {current_loss:.4f}")
            log_fn({'loss': current_loss, 'step': self.step}) # Your logging callback

            # --- TensorBoard Logging ---
            self.writer.add_scalar('loss/train', current_loss, self.step)
            # Get LR from the schedule
            current_lr = self.lr_schedule(self.step)
            self.writer.add_scalar('lr/train', current_lr, self.step)

            # --- Checkpointing ---
            if self.step > 0 and self.step % self.checkpoint_every_steps == 0:
                logging.info(f"Step: {self.step} | Saving checkpoint...")
                try:
                    # Get state from device to save. For pjit-replicated state, jax.device_get is sufficient.
                    state_to_save = jax.device_get(self.state)
                    logging.warning("Checkpoint saving needs adaptation for NnxTrainState!") # This warning remains as per original
                    # Placeholder using old save for ema_params until adapted
                    save_checkpoint(self.ckpt_manager, state_to_save.params, state_to_save.ema_params, self.step)
                except Exception as e:
                    logging.error(f"Error saving checkpoint at step {self.step}: {e}")

            self.step += 1

            # --- Periodic Profiler Trace Flushing ---
            if self.profile_flush_step > 0 and self.step == self.profile_flush_step:
                jax.profiler.stop_trace()
        # --- End of Training Loop ---
        
        # Update self.key with the final state after the loop
        self.key = key

        logging.info('Training completed!')
        # Save final checkpoint
        logging.info("Saving final checkpoint...")
        try:
            # Get final state from device. For pjit-replicated state, jax.device_get is sufficient.
            final_state_to_save = jax.device_get(self.state)
            logging.warning("Final checkpoint saving needs adaptation for NnxTrainState!") # This warning remains as per original
            # Placeholder using old save for ema_params until adapted
            save_checkpoint(self.ckpt_manager, final_state_to_save.params, final_state_to_save.ema_params, self.step)
        except Exception as e:
            logging.error(f"Error saving final checkpoint at step {self.step}: {e}")

        # Close TensorBoard writer
        self.writer.close()
        logging.info(f"TensorBoard logs saved to: {self.tensorboard_dir}")
        logging.info(f"View TensorBoard with: tensorboard --logdir={self.tensorboard_dir}")