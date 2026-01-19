import os

from omegaconf import OmegaConf


def create_info_file(cfg, model_path):
    return OmegaConf.save(config=cfg, f=model_path)


def track_weights(cfg, job_id):
    model_path = os.path.join(cfg.paths.ckpt_dir, str(job_id) + ".yaml")
    create_info_file(cfg, model_path)
    return
