# Table 3 — Distance Correlation in Magnitude-Limited Samples

Measures the dependence between embeddings and redshift using distance correlation (dcorr) across magnitude-limited subsamples.

## Files

| File | Description |
|---|---|
| `table_3.ipynb` | Aggregates dcorr results and formats the table |
| `dcorr_mag_values_counts.py` | Computes distance correlation and sample counts per magnitude bin |
| `dcorr_mag_values_counts.sh` | Shell script to run `dcorr_mag_values_counts.py` across multiple magnitude limits |

## What it measures

Distance correlation (dcorr, via the `dcor` package) quantifies the statistical dependence between an embedding and redshift `z` — unlike linear correlation, it detects arbitrary dependencies. The analysis is repeated across magnitude-limited galaxy subsamples to test whether disentanglement holds under different selection functions.

## Reproducing the table

1. Run `dcorr_mag_values_counts.sh` to compute dcorr values (these are saved as CSV files).
2. Open `table_3.ipynb` and run all cells to aggregate and format the results.
