# Table 2 — Parameter Regression Benchmark

Bootstrap R² results for predicting galaxy physical parameters from each embedding type.

## Files

| File | Description |
|---|---|
| `parameter_regression.ipynb` | Main notebook; loads CSVs and formats the LaTeX tables |
| `model_bootstrap_results.csv` | Bootstrap R² results (with scaling applied) |
| `model_bootstrap_results_no_scaling.csv` | Bootstrap R² results (without input scaling) |
| `r2_results_table.tex` | Generated LaTeX table (full parameter set) |
| `split_r2_tables.tex` | Generated LaTeX table (split by parameter group) |

## What it measures

For each embedding type (`orig`, `cond`, `uncond`) and each galaxy property (e.g. stellar mass `logM*`, SFR `logSFR`, dust attenuation `A_v`), a regression model is trained and evaluated via bootstrap resampling. The reported metric is mean ± 95% CI R² across bootstrap samples.

The table reveals which physical information is retained in (or disentangled from) each embedding.

## Reproducing the table

Run `parameter_regression.ipynb` top-to-bottom. The notebook reads from the pre-computed CSV files and outputs LaTeX source.
