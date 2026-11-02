# Train the flows used by the symbolic-regression paper

The saved-result analysis lives in
[wwdc_symbolic_regression](https://github.com/Birr0/wwdc_symbolic_regression).
This repository contains the neural training and embedding code. `paper.py`
provides the two paper recipes without requiring Hydra launchers or W&B credentials.

## Inputs

Download the `full` data profile from the reproduction repository:

```bash
uv sync --frozen --extra catalogues
uv run --frozen --extra catalogues python download_data.py --profile full --outdir data/paper
```

The HF release must have been uploaded and its immutable revision committed first.
See the reproduction repository's `docs/DATA_RELEASE.md` for publication status.
Do not create a new random spectrum split. The downloader verifies the archived
files and preserves parent counts 422,481 / 52,810 / 52,811 and row order.

Use the downloaded `data/paper/data` directory as `--data-root` below.
The adjacent `pretrained/torch` directory supplies the exact Spender encoder cache;
`paper.py` sets `TORCH_HOME` to that directory unless it is already configured.
The normalisation cache is also included in the full download. Its keys bind it
to preprocessing, wavelength/mask grids and dataset fingerprints.

## Train

From this repository, with Python 3.12 and uv:

```bash
uv run --script --locked paper.py train \
  --arm redshift --data-root /path/to/data/paper/data --outdir outputs/redshift

uv run --script --locked paper.py train \
  --arm norm --data-root /path/to/data/paper/data --outdir outputs/norm
```

These are full GPU training runs. They use seed 42 by default, a frozen
pretrained Spender encoder, ten latent coordinates, hidden width 1024, three
flow layers, AdamW at 0.0001 and batch size 512. The maximum is 100 epochs;
early stopping uses validation loss, patience 10 and minimum improvement 0.0001.
The trainer settings and NORM statistics come from the archived run YAML files,
whose hashes are recorded in `src/conf/paper/{redshift,norm}.json`.

The NORM context is training-standardised log10 of the actual spectral
preprocessing normalisation. Its reference mean/std and retained training count
are checked. Training statistics are not fitted on validation or test rows.
The redshift recipe conditions on raw z. Historical three-seed symbolic searches
reuse these flow representations; they are not three independently trained flows.

Outputs include `ckpts/best.ckpt`, `ckpts/last.ckpt`, CSV metrics and `training.json`.
The latter records the recipe, effective trainer settings, source hashes,
normalisation statistics and run status. Testing uses the best validation
checkpoint. CSV logging replaces the original machine-specific W&B configuration.
Use a new output directory for each run.

To validate parent counts and inspect the recipe without training, append `--check`.
For a one-batch CPU train/validation/test check, use a new output directory and
append `--device cpu --workers 0 --smoke`. Smoke checkpoints are not paper models.

## Generate embeddings

For a newly trained flow:

```bash
uv run --script --locked paper.py embed \
  --arm norm --data-root /path/to/data/paper/data \
  --checkpoint outputs/norm/ckpts/best.ckpt --outdir outputs/norm-embeddings
```

For exact saved models, point `--checkpoint` at the downloaded archive:

| Arm | Checkpoint relative to `--data-root` |
|---|---|
| redshift | `sdss_II/spender_I_flow_v2/spender_I_flow_v2/ckpts/7655991_0.ckpt` |
| norm | `sdss_II/spender_I_normalisation_flow/ckpts/norm_20261012183217_0.ckpt` |

Embedding uses CPU by default and midpoint integration with 20 steps; use
`--device cuda` for GPU inference. It writes `train/`, `val/`, `test/` parquet
shards with spectrum IDs, orig/cond/uncond coordinates, context, redshift and mask
ratio. `embedding_manifest.json` identifies the checkpoint and partial outputs.
Append `--limit 32 --batch-size 32` for an initial check; omit `--limit` for all rows.

## What is verified

CPU one-batch training/validation/test checks pass for both recipes. Checkpoint
inference was checked on the first 32 rows of each split for each arm. NORM agrees
with archived embeddings to sub-micro precision. Redshift embeddings for rows
passing the paper's fixed `mask_ratio <= 0.5` cut agree within the historical
2e-6 / 2e-5 / 2e-5 tolerances for orig/cond/uncond. Two fully masked rows in the
train/validation checks differ more; these are excluded by the pre-existing
paper mask cut and remain present in the released raw data.

Full flow retraining and full-dataset inference have not been rerun. The archived
YAMLs establish the recorded configurations; the current observed source is not
proof of the exact historical training source. Retraining need not yield
bit-identical weights or the same symbolic equations. Use the released checkpoints
and embeddings when reproducing the numerical results. The verification record
is `docs/paper_verification.json`.

The separate `paper.py.lock` reuses the locked analysis/training environment from
the reproduction project. It is independent of the broad exploratory dependencies
in the repository's existing `pyproject.toml`. Unit tests use the source under
`src/`: `PYTHONPATH=src python -m unittest discover -s tests -v` in that environment.
