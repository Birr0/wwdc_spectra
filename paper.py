# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.4.4",
#   "pandas==3.0.2",
#   "scikit-learn==1.8.0",
#   "sympy==1.14.0",
#   "scipy==1.17.1",
#   "matplotlib==3.10.9",
#   "scienceplots==2.2.1",
#   "astroML==1.0.2.post1",
#   "astropy==7.2.0",
#   "pyarrow==24.0.0",
#   "PyYAML==6.0.3",
#   "tqdm==4.67.3",
#   "pyoperon==0.4.0",
#   "torch==2.11.0",
#   "torchvision==0.26.0",
#   "lightning==2.6.1",
#   "spender==0.2.9",
#   "flow-matching==1.0.10",
#   "torchdiffeq==0.2.5",
#   "datasets==4.8.5",
#   "timm==1.0.26",
#   "python-dotenv==1.2.2",
# ]
# ///
"""Train paper flows or generate embeddings using the archived spectrum splits."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PARENT_ROWS = {"train": 422481, "val": 52810, "test": 52811}


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recipe(arm):
    return json.loads((ROOT / f"src/conf/paper/{arm}.json").read_text())


def prepare(args):
    os.environ["DATA_ROOT"] = str(args.data_root.resolve())
    os.environ.setdefault("TORCH_HOME", str(args.data_root.resolve().parent / "pretrained/torch"))
    sys.path.insert(0, str(ROOT / "src"))
    saved = args.data_root / "sdss/sdss_II_catalog"
    if not (saved / "dataset_dict.json").is_file():
        raise ValueError(f"Exact archived splits required at {saved}; this entry point never recreates splits")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.outdir}")
    from datasets import load_from_disk
    archive = load_from_disk(str(saved))
    for split, count in PARENT_ROWS.items():
        if len(archive[split]) != count:
            raise ValueError(f"{split}: expected {count} archived parent rows, got {len(archive[split])}")
    return recipe(args.arm)


def make_datasets(settings, workers):
    from wwdc_spectra.data.sdss import SDSS
    catalog = {"variables": settings["variables"], "drop_variables": []}
    datasets = {split: SDSS(split=split, y_catalog=catalog, normalisation_workers=workers)
                for split in PARENT_ROWS}
    reference = settings.get("normalisation_reference")
    if reference:
        import numpy as np
        for dataset in datasets.values():
            np.testing.assert_allclose([dataset.normalisation_stats[k] for k in ("mean", "std")],
                                       [reference[k] for k in ("mean", "std")], rtol=0, atol=1e-12)
        for key in ("parent_rows", "retained_rows", "excluded_rows"):
            if datasets["train"].normalisation_stats[key] != reference[key]:
                raise ValueError(f"Training NORM membership differs: {key}")
    return catalog, datasets


def make_model(settings, catalog, checkpoint=None):
    import torch
    from wwdc_spectra.models.load_spender import PretrainedSpender
    from wwdc_spectra.models.model import LightningFlowMatching, WrappedModel
    model = LightningFlowMatching(base_model=PretrainedSpender("sdss_I", 10),
        lr=settings["lr"], batch_size=settings["batch_size"], code_dim=settings["code_dim"],
        hidden_dim=settings["hidden_dim"], n_layers=settings["n_layers"], catalog=catalog, n_steps=20)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        from flow_matching.solver import ODESolver
        model.wrapped_vf = WrappedModel(model.vf)
        model.solver = ODESolver(velocity_model=model.wrapped_vf)
    return model


def train(args, settings):
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader
    L.seed_everything(args.seed, workers=True)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    workers = args.workers if args.workers is not None else settings["num_workers"]
    catalog, datasets = make_datasets(settings, workers)
    model = make_model(settings, catalog)
    loaders = {split: DataLoader(dataset, batch_size=settings["batch_size"] if not args.smoke else 4,
                                shuffle=split == "train", num_workers=workers)
               for split, dataset in datasets.items()}
    args.outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=args.outdir / "ckpts", filename="best", monitor="val_loss", mode="min", save_last=True)
    options = {k: v for k, v in settings["trainer"].items() if k != "_target_"}
    options.update(accelerator=args.device, devices=1)
    if args.device == "cpu":
        options["precision"] = "32-true"
    if args.smoke:
        options.update(max_epochs=1, limit_train_batches=1, limit_val_batches=1, limit_test_batches=1,
                       num_sanity_val_steps=0, log_every_n_steps=1)
    manifest = {"status": "running", "arm": args.arm, "seed": args.seed, "smoke": args.smoke,
                "recipe": settings, "effective_trainer": options,
                "normalisation": {split: getattr(dataset, "normalisation_stats", None) for split, dataset in datasets.items()},
                "source": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "src/wwdc_spectra").rglob("*.py"))},
                "notes": "CSV logging replaces historical W&B paths; retraining is not a claim of bit-identical historical weights."}
    manifest_path = args.outdir / "training.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        trainer = L.Trainer(**options, logger=CSVLogger(str(args.outdir), name="metrics"),
                            callbacks=[checkpoint, EarlyStopping(**settings["early_stopping"])])
        trainer.fit(model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
        scores = trainer.test(model, dataloaders=loaders["test"], ckpt_path="best")
        manifest.update(status="passed", checkpoint=checkpoint.best_model_path, test=scores)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def embed(args, settings):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import DataLoader, Subset
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    catalog, datasets = make_datasets(settings, args.workers or 0)
    model = make_model(settings, catalog, args.checkpoint).to(args.device).eval()
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "checkpoint_sha256": file_hash(args.checkpoint), "arm": args.arm,
                "recipe": settings, "runner_sha256": file_hash(Path(__file__)),
                "partial": args.limit is not None, "method": "midpoint", "steps": 20, "splits": {}}
    path = args.outdir / "embedding_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for split, dataset in datasets.items():
            dataset.return_id = dataset.return_pos = dataset.return_mask_ratio = True
            selected = Subset(dataset, range(min(args.limit, len(dataset)))) if args.limit else dataset
            folder = args.outdir / split
            folder.mkdir()
            rows = 0
            for index, batch in enumerate(DataLoader(selected, batch_size=args.batch_size, shuffle=False, num_workers=args.workers or 0)):
                X, y, spec_id, object_id, ra, dec, z, mask_ratio = batch
                with torch.inference_mode():
                    pred = model.predict_step(X.to(args.device), y.to(args.device), ["orig", "cond", "uncond"])
                values = {key: value.detach().cpu().tolist() for key, value in pred.items()}
                values.update(id=np.asarray(spec_id), BESTOBJID=np.asarray(object_id), ra=np.asarray(ra),
                              dec=np.asarray(dec), z=np.asarray(z), mask_ratio=np.asarray(mask_ratio), y=y.tolist())
                pq.write_table(pa.table(values), folder / f"{index}.parquet")
                rows += len(X)
            manifest["splits"][split] = rows
        manifest["status"] = "passed"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "embed"])
    parser.add_argument("--arm", choices=["redshift", "norm"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, help="Partial embedding check only; omit for full splits")
    parser.add_argument("--smoke", action="store_true", help="One train/validation/test batch; not a paper training run")
    parser.add_argument("--check", action="store_true", help="Validate archived parent counts and display the recipe without fitting")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1) or (args.workers is not None and args.workers < 0):
        parser.error("Batch size and limit must be positive; workers must be nonnegative")
    if args.stage == "embed" and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error("embed requires an existing --checkpoint")
    args.device = args.device or ("gpu" if args.stage == "train" else "cpu")
    if args.stage == "embed" and args.device == "gpu":
        args.device = "cuda"
    elif args.stage == "train" and args.device == "cuda":
        args.device = "gpu"
    settings = prepare(args)
    if args.check:
        print(json.dumps(settings, indent=2))
        return
    (train if args.stage == "train" else embed)(args, settings)


if __name__ == "__main__":
    main()
