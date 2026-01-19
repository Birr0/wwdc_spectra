import logging
import os

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from modules import track_weights

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="experiment/spender_I_flow/train",
)
def main(cfg):
    """The main training function."""
    seed_everything(cfg.seed, workers=True)

    try:
        data = hydra.utils.instantiate(cfg.data.loader)
        data.setup()
        train_dataloader = data.train_dataloader()
        val_dataloader = data.val_dataloader()
        test_dataloader = data.test_dataloader()

        log.info("Data loaders initialized.")
    except Exception as e:
        msg = f"Error in instantiating the data loader: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        if "run_id" in cfg:
            msg = f"Resuming training from run_id: {cfg.run_id}."
            log.info(msg)
            if os.path.exists(f"{cfg.paths.experiment_path}/ckpts/{cfg.run_id}.ckpt"):
                OmegaConf.update(
                    cfg,
                    "lightning_loader.vae_ckpt_path",
                    f"{cfg.paths.experiment_path}/ckpts/{cfg.run_id}.ckpt",
                )
        lightning_loader = hydra.utils.instantiate(cfg.lightning_loader)
        log.info("Lightning loader initialized.")
    except Exception as e:
        msg = f"Error in instantiating the lightning loader: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        trainer = hydra.utils.instantiate(cfg.trainer)
        log.info("Trainer initialized.")
    except Exception as e:
        msg = f"Error in instantiating the trainer: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        if "run_id" in cfg:
            OmegaConf.update(cfg, "logger.wandb.id", f"{cfg.run_id}")
        wandb = hydra.utils.instantiate(cfg.logger.wandb)
        if "run_id" in cfg:
            wandb._wandb_init["resume"] = "must"
        wandb.log_hyperparams(cfg)
        log.info("Wandb logger initialized.")
    except Exception as e:
        msg = f"Error in instantiating the wandb logger: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        trainer.fit(
            model=lightning_loader,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.trainer_ckpt_path if cfg.trainer_ckpt_path else None,
        )
        log.info("Model fitting completed.")

    except Exception as e:
        msg = f"Failed to train the model for job: {HydraConfig.get().job.id}. \
         The following is the cause of the error: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        trainer.test(
            model=lightning_loader,
            dataloaders=test_dataloader,
        )
        log.info("Model testing completed.")

    except Exception as e:
        msg = f"Failed to test the model for job: {HydraConfig.get().job.id}. \
         The following is the cause of the error: {e}."
        log.error(msg)
        raise Exception(msg) from e

    try:
        job_id = HydraConfig.get().job.id
        track_weights(cfg, job_id)
        log.info("Weights tracking on local machine completed.")
    except Exception as e:
        msg = f"Failed to track the weights for job: {HydraConfig.get().job.id}. \
         The following is the cause of the error: {e}."
        log.error(msg)
        raise Exception(msg) from e

    return


if __name__ == "__main__":
    # Enable multirun by default for SLURM launcher
    GlobalHydra.instance().clear()
    main()
