"""Exact SDSS normalisation cache and training-only conditioning statistics."""
import hashlib
import inspect
import logging
import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class NormalisationRows(Dataset):
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source.dataset)

    def __getitem__(self, index):
        row = self.source.dataset[index]
        sp = row["spectrum"]
        _, _, norm, _, _ = self.source.prepare_spectrum(
            sp["flux"], sp["ivar"], sp["lambda"], sp["mask"], row["Z"])
        return index, float(norm)


def cached_norms(source, cache_dir, workers):
    # Include split fingerprint, preprocessing implementation and wavelength/mask
    # grids: a stale cache must not silently change membership or train scaling.
    h = hashlib.sha256(inspect.getsource(source.prepare_spectrum).encode())
    h.update(source.dataset._fingerprint.encode())
    h.update(source._wave_obs.numpy().tobytes())
    h.update(source._skyline_mask.numpy().tobytes())
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (h.hexdigest() + ".npy")
    if path.exists():
        values = np.load(path, allow_pickle=False)
        if values.shape != (len(source.dataset),):
            raise ValueError(f"Invalid normalisation cache shape: {path}")
        return values, str(path)
    values = np.full(len(source.dataset), np.nan, dtype=np.float64)
    loader = DataLoader(NormalisationRows(source), batch_size=512,
                        num_workers=workers, shuffle=False)
    for batch, (positions, norms) in enumerate(loader):
        values[positions.numpy()] = norms.numpy()
        if batch % 100 == 0:
            log.info("Computing exact normalisation: %d/%d", min((batch+1)*512, len(values)), len(values))
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.save(stream, values)
    os.replace(temp, path)
    return values, str(path)


def fit_log_scaling(norms):
    valid = np.isfinite(norms) & (norms > 0)
    x = np.log10(norms[valid])
    if len(x) < 2 or not np.isfinite(x.std()) or x.std() <= 0:
        raise ValueError("Need at least two varying positive training normalisations")
    return float(x.mean()), float(x.std())


def configure_normalisation(source, split, cache_dir, workers):
    from wwdc_spectra.data.sdss import DATA_DIR, SDSS
    cache_dir = Path(cache_dir) if cache_dir else Path(DATA_DIR).parent / "normalisation_cache"
    # This plain training source does not itself request NORM, avoiding recursion.
    training = source if split == "train" else SDSS(split="train", y_catalog=None)
    train_norms, train_path = cached_norms(training, cache_dir, workers)
    mean, std = fit_log_scaling(train_norms)
    values, path = (train_norms, train_path) if split == "train" else cached_norms(source, cache_dir, workers)
    valid = np.isfinite(values) & (values > 0)
    source.normalisation_stats = {
        "mean": mean, "std": std, "transform": "standardised log10", "fit_split": "train",
        "parent_rows": len(values), "retained_rows": int(valid.sum()),
        "excluded_rows": int((~valid).sum()), "cache": path, "training_cache": train_path,
    }
    log.info("NORM %s: %s", split, source.normalisation_stats)
    source.dataset = source.dataset.select(np.flatnonzero(valid).tolist())
