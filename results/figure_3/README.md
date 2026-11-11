# Figure 3 — Redshift Correlation in Magnitude-Limited Sample

Visualizes how redshift correlations in the embeddings vary across magnitude-limited galaxy subsamples.

## Files

| File | Description |
|---|---|
| `z_correlation_mag_limited_v2.ipynb` | Main figure notebook |
| `dcorr_mag_values_v2.py` | Computes distance correlation values per magnitude bin |
| `dcorr_mag_values_v2.sh` | Shell script to run `dcorr_mag_values_v2.py` across multiple configurations |

## What it shows

Plots distance correlation between each embedding type (`orig`, `cond`, `uncond`) and spectroscopic redshift `z` as a function of apparent magnitude limit.

## Reproducing the figure

1. Run `dcorr_mag_values_v2.sh` to compute distance correlations at each magnitude limit (outputs CSV files).
2. Open `z_correlation_mag_limited_v2.ipynb` and run all cells to generate the figure.
