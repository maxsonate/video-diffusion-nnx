import argparse
import yaml
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import logging
from flax import nnx
from unet3d import Unet3D
from gaussian_diffusion import GaussianDiffusion
from utils import load_checkpoint, video_array_to_gif
from einops import rearrange

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Parses arguments, loads config, initializes components, and generates samples."""
    parser = argparse.ArgumentParser(description='Generate samples using diffusion model')
    parser.add_argument(
        '--config',
        type=str,
        default=str(Path(__file__).parent / 'configs' / 'config.yaml'),
        help='Path to the YAML config file'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=str(Path(__file__).parent / 'outputs'),
        help='Directory to save sampled GIFs'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        required=True,
        help='Path to the model checkpoint file'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=0,
        help='Checkpoint step number to load'
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output_path)
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

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

    checkpoint_path = Path(args.checkpoint_path).resolve()
    diffusion_model = load_checkpoint(diffusion_model, args.step, str(checkpoint_path))
    logging.info(f"Loaded checkpoint from {checkpoint_path} at step {args.step}")

    sampled_videos = diffusion_model.sample()
    logging.info(f"Sampled {len(sampled_videos)} videos")

    # Rearrange dimensions and normalize to [0,255]
    sampled_videos = rearrange(sampled_videos, 'b c f h w -> b f h w c')
    min_val = jnp.min(sampled_videos)
    max_val = jnp.max(sampled_videos)
    normalized = (sampled_videos - min_val) / (max_val - min_val)
    uint8_videos = (normalized * 255).astype(jnp.uint8)

    # Convert to NumPy and save as GIF
    video_array_to_gif(np.array(uint8_videos[1]), output_path / 'test_vid.gif')

if __name__ == "__main__":
    main()









    