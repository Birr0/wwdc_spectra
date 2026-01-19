import logging
from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch import seed_everything
import pyarrow as pa
import pyarrow.parquet as pq

from wwdc_spectra.inference.modules import (
    create_lightning_loader
)
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
        msg = "Lightning loader instantiated."
        log.info(msg)
    except Exception as e:
        msg = f"Error instantiating lightning loader: {e}"
        log.error(msg)
        raise e

    for name, split in split_dataloaders.items():
        if split["call"]:
            for idx, batch in enumerate(split["dataloader"]):
                if (cfg.batch_limit is not None) and (idx >= cfg.batch_limit):
                    break
                try:
                    # need to add embed_opt to the config.
                    X, y, _id = batch
                    predictions = lightning_loader.predict_step(
                        X=X, y=y, embed_opt=cfg.embed_opt
                    )

                    data = {k: v.tolist() for k, v in predictions.items()}
                    data["id"] = _id

                    path = Path(cfg.paths.embed_dir + f"{model_id}/{name}/")
                    if not path.exists():
                        path.mkdir(parents=True, exist_ok=True)

                    pq.write_table(
                        pa.table(data), 
                        path / f"{idx}.parquet"
                    )

                except Exception as e:
                    msg = f"Error creating embeddings: {e}"
                    log.error(msg)
                    raise e

            try:
                cfg.logger.wandb.tags.append(str(model_id))
                cfg.logger.wandb.id = model_id
                cfg.logger.wandb.name = f"{model_id}_{cfg.meta.experiment_name}"
                #wandb_logger = hydra.utils.instantiate(cfg.logger.wandb)
                msg = "Wandb logger instantiated."
                log.info(msg)

            except Exception as e:
                msg = f"Error instantiating wandb logger: {e}"
                log.error(msg)
                raise e
            
            '''
            #Don't include wandb logging
            try:
                print(f"wandb_logger: {wandb_logger}")
                print(
                    f"wandb_logger._wandb_init['mode']:\
                        {wandb_logger._wandb_init['mode']}"
                )
                if wandb_logger._wandb_init["mode"] == "online":
                    wandb_logger.log_table(
                        key=f"{model_id}_{name}_embeddings",
                        dataframe=wandb_format(embeddings, cfg.data.x_ds),
                    )
                    print("Logged embeddings to wandb.")
                    msg = "Logged embeddings to wandb."
                    log.info(msg)
            except Exception as e:
                msg = f"Error logging embeddings to wandb: {e}"
                log.error(msg)
                raise e
            '''

    log.info("Inference complete.")
    return


if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
