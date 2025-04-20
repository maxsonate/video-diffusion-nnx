# Video Diffusion Model (JAX/Flax NNX)

This project implements a video diffusion model using JAX and the Flax NNX library. It allows for training a model to generate video sequences based on the principles of denoising diffusion probabilistic models.

## Setup

Ensure you have the necessary dependencies installed (e.g., `jax`, `flax`, `optax`, `orbax-checkpoint`, `torch`, `torchvision`, `einops`, `PyYAML`). You might use a `requirements.txt` or manage dependencies manually.

```bash
# Example dependency installation (adjust as needed)
pip install -r requirements.txt
```

## Training

Training is configured via a YAML file (default: `configs/config.yaml`). Modify this file to set hyperparameters, dataset paths, model dimensions, etc.

To start training, run:

```bash
python train.py --config path/to/your/config.yaml
```

*   Replace `path/to/your/config.yaml` with the actual path to your configuration file if it's not the default.
*   Checkpoints (including model and optimizer states) will be saved according to the settings in the config file (e.g., `checkpoint_dir_path`, `checkpoint_every_steps`, `max_to_keep`).
*   To resume training from a specific step, you can either:
    *   Modify the `resume_training_step` value within the `trainer` section of your config file.
    *   Or, override the config value using the `--resume_step` command-line argument:
        ```bash
        python train.py --config path/to/config.yaml --resume_step <step_number>
        ```

## Sampling

To generate video samples from a trained checkpoint:

```bash
python sample.py --checkpoint-path /path/to/checkpoint/dir --step <step_number> --output-path /path/to/save/output --config /path/to/config.yaml
```

**Arguments:**

*   `--checkpoint-path`: Path to the **directory** containing the saved checkpoints (e.g., `./results/checkpoints`).
*   `--step`: The specific training step number of the checkpoint to load.
*   `--output-path`: The directory where the generated sample GIF(s) will be saved (e.g., `./samples`).
*   `--config`: Path to the YAML configuration file that matches the model architecture used for the checkpoint (usually the same one used for training).

**Example:**

```bash
python sample.py --checkpoint-path ./results/checkpoints --step 50000 --output-path ./generated_samples --config configs/config.yaml
```
