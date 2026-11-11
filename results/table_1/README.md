# Table 1 — Redshift Prediction Benchmark

Evaluates how much redshift information is retained in, or removed from, each embedding type using linear and MLP probes.

## Files

| File | Description |
|---|---|
| `table_1.ipynb` | Aggregates results from all CSV files and prints the summary table |
| `z_linear_prediction.py` | Fits a linear probe (OLS) on each embedding type to predict redshift `z` |
| `z_linear_prediction.sh` | Shell script to run `z_linear_prediction.py` across multiple models/splits |
| `z_mlp_prediction_v2.py` | Fits a 3-layer MLP probe on each embedding type to predict `z` |
| `z_mlp_prediction_v2.sh` | Shell script to run `z_mlp_prediction_v2.py` across multiple models/splits |

## What it measures

For each base model (Spender I, Spender II) and each embedding type (`orig`, `cond`, `uncond`), a probe is trained to predict spectroscopic redshift `z` from the embedding. The reported metric is R² on a held-out test set, averaged over bootstrap resamples.

Lower R² for `uncond` indicates more redshift information has been removed by the conditional flow transport.

## Reproducing the table

1. Run the prediction scripts (or their shell wrappers) to generate CSV files in subdirectories named `linear_predictions_<model>_v2/` and `mlp_predictions_<model>_v2/`.
2. Open `table_1.ipynb` and run all cells.
