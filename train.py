"""Main training script for the video diffusion model."""

import argparse
import yaml
from pathlib import Path

from flax import nnx
from unet3d import Unet3D
from gaussian_diffusion import GaussianDiffusion
from trainer import Trainer


def main():
    """Parses arguments, loads config, initializes components, and starts training."""
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
        default=None,
        help='Step to resume training from'
    )
    args = parser.parse_args()
    config_path = Path(args.config)

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Instantiate Unet3D from config
    unet_cfg = config['unet']
    print("Initializing Unet3D model...")
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
    print("Initializing GaussianDiffusion model...")
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
    print("Initializing Trainer...")
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
        max_to_keep=trainer_cfg.get('max_to_keep', None)
    )

    # TODO: Add loading from checkpoint before starting training

    print("Starting training...")
    trainer.train()
    print("Training finished.")


if __name__ == '__main__':
    main() 