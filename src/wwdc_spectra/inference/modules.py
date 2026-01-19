import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

def convert_to_np(values):
    return [value.detach().cpu().numpy() for value in values]

def create_embeddings(predictions, split):
    _catalog = {}

    for col, value in predictions["catalog"].items():
        _catalog[col] = convert_to_np(value)

    _catalog["z"] = convert_to_np(predictions["z"])

    predictions.pop("catalog")
    predictions.update(_catalog)

    """
    # To-Do: features for hugging face.
    Put in x_ds.
    features = {
        "X": Sequence(feature=Value("float32")),
        "y": Sequence(feature=Value("float32")),
        "z": Sequence(feature=Value("float32")),
        "z_prime": Sequence(feature=Value("float32")),
    }

    for col in _catalog.keys():
        features[col] = Sequence(feature=Value("float32"))
    """

    del _catalog

    return Dataset.from_dict(predictions, split=split)

def wandb_format(embeddings, x_ds):
    X_collection = []
    recon_collection = []
    z_collection = []

    if x_ds["type"] == "image":
        for X in embeddings["X"]:
            X_collection.append(wandb.Image(torch.tensor(X)))

        if "recon" in embeddings.column_names:
            for recon in embeddings["recon"]:
                recon_collection.append(wandb.Image(torch.tensor(recon)))

    if "z" in embeddings.column_names:
        for z in embeddings["z"]:
            z_collection.append(torch.tensor(z))

    embeddings = embeddings.to_pandas()
    if X_collection:
        embeddings["X"] = X_collection
    if recon_collection:
        embeddings["recon"] = recon_collection
    if z_collection:
        embeddings["z"] = z_collection
    return embeddings

def create_lightning_loader(cfg):
    ckpt_dir = Path(cfg.paths.ckpt_dir)
    ckpt_files = get_ckpt_files(ckpt_dir)
    current_job_num = int(HydraConfig.get().job.num)

    job_ids = []
    job_nums = []

    for ckpt_file in ckpt_files:
        if len(ckpt_file.stem.split("_")) > 1:
            job_id, job_num = ckpt_file.stem.split("_")

        else:
            job_id = ckpt_file.stem
            job_num = 0  # for single job runs.

        if "-v" in job_num:
            # skip versioned files
            continue

        job_ids.append(job_id)
        job_nums.append(int(job_num))

    job_ids = set(job_ids)

    if len(job_ids) > 1:
        msg = (
            f"Multiple job ids found in {ckpt_dir}. "
            "Only one job id should exist for this run."
        )
        raise ValueError(msg)

    try:
        ckpt_idx = job_nums.index(current_job_num)
    except Exception as e:
        msg = f"Job number {current_job_num} not found in {ckpt_dir}. {e}"
        raise ValueError(msg) from e

    ckpt_path = Path(ckpt_files[ckpt_idx])
    OmegaConf.update(cfg, "lightning_loader.ckpt_path", str(ckpt_path))

    return (
        cfg,
        ckpt_path.stem,
        int(HydraConfig.get().job.id),
    )

def get_ckpt_files(ckpt_dir):
    if not ckpt_dir.exists():
        msg = f"Checkpoint directory {ckpt_dir} does not exist."
        raise FileNotFoundError(msg)

    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    # get idx, job_id and job_num from the ckpt_files
    # and sort by job_num

    if len(ckpt_files) == 0:
        msg = f"No checkpoint files found in {ckpt_dir}."
        raise FileNotFoundError(msg)
    return ckpt_files
