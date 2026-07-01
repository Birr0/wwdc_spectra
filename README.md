# wwdc-spectra

<!-- Add your project description here -->

## Overview

<!-- Describe the goal and motivation of this project -->

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --group dev
```

**Requirements:** Python >= 3.12

### Key Dependencies

| Package | Purpose |
|---|---|
| `spender` | Pretrained galaxy spectrum autoencoder (backbone) |
| `flow-matching` | Flow matching training library |
| `lightning` | PyTorch training framework |
| `hydra-core` | Configuration management |
| `datasets` | HuggingFace dataset loading |
| `wandb` | Experiment tracking |

## Project Structure

```
wwdc_spectra/
├── src/
│   ├── wwdc_spectra/
│   │   ├── data/         # Dataset classes and dataloaders
│   │   ├── models/       # Flow matching models and spectrum encoders
│   │   ├── training/     # Training entry point
│   │   └── inference/    # Embedding generation (inference)
│   └── conf/             # Hydra configuration files
└── results/
    ├── figure_1/         # t-SNE embedding visualization
    ├── figure_2/         # SDSS main galaxy sample analysis
    ├── figure_3/         # Redshift correlation in magnitude-limited sample
    ├── table_1/          # Redshift prediction benchmark (R²)
    ├── table_2/          # Parameter regression benchmark (bootstrap R²)
    └── table_3/          # Distance correlation analysis
```

## Pipeline

### 1. Data

SDSS galaxy spectra are loaded via `src/wwdc_spectra/data/sdss.py`. The dataset is automatically downloaded from the [MultimodalUniverse](https://github.com/MultimodalUniverse/MultimodalUniverse/tree/main/scripts/sdss) scripts if not found locally. Set `DATA_ROOT` in a `.env` file to point to your data directory.

### 2. Training

Training is driven by Hydra. The entry point is `src/wwdc_spectra/training/train.py`:

```bash
srun python src/wwdc_spectra/training/train.py -cn "experiment/{experiment_name}/train" hydra/launcher={launch_config}
```

The flow matching model learns a conditional velocity field over the latent space of a pretrained Spender encoder. 

| Model | Description |
|---|---|
| `LightningFlowMatching` | Standard conditional flow matching with CFG |


### 3. Inference (Embedding Generation)

Once trained, embeddings are extracted with `src/wwdc_spectra/inference/embed.py`:

```bash
srun python src/wwdc_spectra/inference/embed.py -cn "experiment/{experiment_name}/train" hydra/launcher={launch_config}
```

This saves parquet files containing original (`orig`), conditional (`cond`), and unconditional (`uncond`) embeddings for downstream analysis.

### 4. Results

Jupyter notebooks in `results/` reproduce all figures and tables using the saved embeddings.

## Configuration

All experiment configuration lives under `src/conf/`. Key config groups:

- `data/` — dataset and dataloader settings
- `trainer/` — Lightning trainer (CPU, GPU, multi-GPU, MPS, SLURM)
- `logger/` — logging backends (W&B, CSV, checkpoint)
- `optimizer/` — optimizer and early stopping settings
- `hydra/launcher/` — job launchers (local, SLURM/ARC)

## Environment Variables

Create a `.env` file in the project root:

```env
DATA_ROOT=/path/to/your/data
HF_USERNAME=your_huggingface_username
```
