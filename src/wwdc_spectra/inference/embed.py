import logging
from pathlib import Path

import torch
import hydra
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch import seed_everything
import pyarrow as pa
import pyarrow.parquet as pq

from wwdc_spectra.inference.modules import create_lightning_loader
# wandb_format

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="experiment/spender_I_flow/embed",
)
def main(cfg):
    """The main inference function."""
    seed_everything(cfg.seed, workers=True)

    try:
        data = hydra.utils.instantiate(cfg.data.loader)
        data.setup()
        split_dataloaders = {
            "test": {"loader": data.test_dataloader, "call": False},
            "train": {"loader": data.train_dataloader, "call": False},
            "val": {"loader": data.val_dataloader, "call": False},
        }

        for split in cfg.splits:
            split_dataloaders[split]["dataloader"] = split_dataloaders[split][
                "loader"
            ]()
            split_dataloaders[split]["call"] = True

        msg = f"Data loaders instantiated: {split_dataloaders!s}"
        log.info(msg)

    except Exception as e:
        msg = f"Error instantiating data loaders: {e}"
        log.error(msg)
        raise e

    try:
        cfg, model_id, job_id = create_lightning_loader(cfg)
        print(f"cfg: {cfg}. Model ID: {model_id}. Job ID: {job_id}")
        lightning_loader = hydra.utils.instantiate(cfg.lightning_loader)
        print(lightning_loader)
        msg = "Lightning loader instantiated."
        log.info(msg)
    except Exception as e:
        msg = f"Error instantiating lightning loader: {e}"
        log.error(msg)
        raise e

    for name, split in split_dataloaders.items():
        if not split["call"]:
            continue

        path = Path(cfg.paths.embed_dir) / str(model_id) / name
        path.mkdir(parents=True, exist_ok=True)

        writer = None

        try:
            for idx, batch in enumerate(split["dataloader"]):
                if cfg.batch_limit is not None and idx >= cfg.batch_limit:
                    break

                X, y, spec_id, obj_id, ra, dec, z, mask_ratio = batch

                with torch.inference_mode():
                    predictions = lightning_loader.predict_step(
                        X=X,
                        y=y,
                        embed_opt=cfg.embed_opt,
                    )

                data = {
                    k: v.detach().cpu().tolist()
                    for k, v in predictions.items()
                }

                data["id"] = (
                    spec_id.detach().cpu().numpy()
                    if hasattr(spec_id, "detach")
                    else spec_id
                )
                data["BESTOBJID"] = (
                    obj_id.detach().cpu().numpy()
                    if hasattr(obj_id, "detach")
                    else obj_id
                )

                # conditioning values as fed to the flow (e.g. standardised NORM)
                data["y"] = y.detach().cpu().tolist()
                data["z"] = z.detach().cpu().numpy()
                data["ra"] = ra.detach().cpu().numpy()
                data["dec"] = dec.detach().cpu().numpy()
                data["mask_ratio"] = mask_ratio.detach().cpu().numpy()

                table = pa.table(data)

                if writer is None:
                    writer = pq.ParquetWriter(
                        path / "embeddings.parquet",
                        table.schema,
                        compression="zstd",
                    )

                writer.write_table(table)

        except Exception as e:
            msg = f"Error creating embeddings for split {name}: {e}"
            log.error(msg)
            raise

        finally:
            if writer is not None:
                writer.close()

    log.info("Inference complete.")
    return


if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
