"""Main training script for the video diffusion model."""

import logging
import argparse
import yaml
from pathlib import Path

from flax import nnx
from unet3d import Unet3D
from gaussian_diffusion import GaussianDiffusion
from trainer import Trainer


def main():
    """Parses arguments, loads config, initializes components, and starts training."""
    # --- Configure Logging ---
    # Set the minimum level to INFO so info messages are displayed
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s', force=True)
    # You can customize the format further, e.g., add timestamps:
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse command-line arguments for config file
    parser = argparse.ArgumentParser(description='Train diffusion model')
    parser.add_argument(
        '--config',
        type=str,
        default=str(Path(__file__).parent / 'configs' / 'config.yaml'),
        help='Path to the YAML config file'
    )
    parser.add_argument(
        '--resume_step',
        type=int,
        default=0,
        help='Step to resume training from'
    )
    parser.add_argument(
        '--rng_seed',
        type=int,
        default=None,
        help='RNG seed to use for training'
    )
    args = parser.parse_args()
    config_path = Path(args.config)

    # Load configuration
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --- RNG Seed ---
    # Use command-line seed if provided, otherwise use config seed, otherwise default to 0
    master_seed = args.rng_seed if args.rng_seed is not None else config.get('rng_seed', 0)
    logging.info(f"Using master RNG seed: {master_seed}")

    # Instantiate Unet3D from config
    unet_cfg = config['unet']
    logging.info("Initializing Unet3D model...")
    rngs = nnx.Rngs(unet_cfg['rngs_seed'])
    unet_model = Unet3D(
        dim=unet_cfg['dim'],
        rngs=rngs,
        dim_mults=tuple(unet_cfg['dim_mults']),
        channels=unet_cfg['channels'],
        use_bert_text_cond=unet_cfg['use_bert_text_cond'],
    )

    # Instantiate GaussianDiffusion from config
    diff_cfg = config['diffusion']
    logging.info("Initializing GaussianDiffusion model...")
    diffusion_model = GaussianDiffusion(
        denoise_fn=unet_model,
        image_size=diff_cfg['image_size'],
        num_frames=diff_cfg['num_frames'],
        timesteps=diff_cfg['timesteps'],
        loss_type=diff_cfg['loss_type'],
        channels=diff_cfg['channels'],
    )

    # Instantiate Trainer from config
    trainer_cfg = config['trainer']
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        diffusion_model=diffusion_model,
        folder=trainer_cfg['folder'],
        dataset_path=trainer_cfg['dataset_path'],
        num_frames=trainer_cfg['num_frames'],
        train_batch_size=trainer_cfg['train_batch_size'],
        train_lr=trainer_cfg['train_lr'],
        train_num_steps=trainer_cfg['train_num_steps'],
        gradient_accumulate_every=trainer_cfg['gradient_accumulate_every'],
        step_start_ema=trainer_cfg['step_start_ema'],
        update_ema_every=trainer_cfg['update_ema_every'],
        save_and_sample_every=trainer_cfg['save_and_sample_every'],
        results_folder=trainer_cfg['results_folder'],
        num_sample_rows=trainer_cfg['num_sample_rows'],
        max_grad_norm=trainer_cfg['max_grad_norm'],
        use_path_as_cond=trainer_cfg['use_path_as_cond'],
        sample_text=trainer_cfg['sample_text'],
        cond_scale=trainer_cfg['cond_scale'],
        checkpoint_every_steps=trainer_cfg['checkpoint_every_steps'],
        checkpoint_dir_path=trainer_cfg['checkpoint_dir_path'],
        add_loss_plot=trainer_cfg['add_loss_plot'],
        resume_training_step=args.resume_step,
        tensorboard_dir=trainer_cfg['tensorboard_dir'],
        max_to_keep=trainer_cfg.get('max_to_keep', None),
        lr_decay_start_step=trainer_cfg['lr_decay_start_step'],
        lr_decay_steps=trainer_cfg['lr_decay_steps'],
        lr_decay_coeff=trainer_cfg['lr_decay_coeff'],
        profile_flush_step=trainer_cfg['profile_flush_step'],
        rng_seed=master_seed,
    )

    # TODO: Add loading from checkpoint before starting training

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training finished.")


if __name__ == '__main__':
    main() 