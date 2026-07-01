# Figure 1 — t-SNE Embedding Visualization

t-SNE visualization of the three embedding types to qualitatively assess the redshift structure of the learned representations.

## Files

| File | Description |
|---|---|
| `tsne_embed.ipynb` | Computes t-SNE projections and produces the figure |

## What it shows

A 2D t-SNE projection of `orig`, `cond`, and `uncond` embeddings colored by galaxy properties (e.g. redshift, stellar mass). The figure illustrates how conditional flow matching reshapes the latent space relative to the original Spender embedding.

## Reproducing the figure

Open `tsne_embed.ipynb` and run all cells. Point the notebook at the embedding parquet files produced by `src/wwdc_spectra/inference/embed.py`.
