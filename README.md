# WWDC spectra

Neural flow training and embedding generation for the WWDC symbolic-regression
paper. The saved-equation analysis and data downloader live in
[wwdc_symbolic_regression](https://github.com/Birr0/wwdc_symbolic_regression).

Start with [the paper training walkthrough](docs/PAPER_TRAINING.md).
It covers the exact spectrum splits, redshift/NORM recipes and checkpoint inference.

With the full data bundle downloaded, train either paper flow:

```bash
uv run --script --locked paper.py train \
  --arm norm --data-root /path/to/data/paper/data --outdir outputs/norm
```

Use `--arm redshift` for the redshift-conditioned flow. Both default to seed 42.
Training requires a GPU by default; a CPU smoke check is available in the walkthrough.
CSV logs and local checkpoints are written without a W&B account.

The paper recipes are in `src/conf/paper/`; the implementation is under
`src/wwdc_spectra/`. The original Hydra training/inference entry points and NORM
configurations remain available in `src/conf/experiment/`.

Full retraining has not been verified. For numerical reproduction, use the released
checkpoints/embeddings and the reproduction repository's saved-result workflow.
