import os

import torch
from dotenv import load_dotenv
from flow import VelocityField

load_dotenv()


data_name = "sdss_II"
experiment_id = "7000609" # 6997867
experiment_name = "spender_II_flow"
model_names = [
    f"{experiment_id}_{i}.ckpt" for i in range(3)
]
data_root = os.getenv("DATA_ROOT")
HF_USERNAME = os.getnev("HF_USERNAME")



def ckpt_path(experiment_name, model_name):
    return f"{data_root}/{data_name}/{experiment_name}/ckpts/{model_name}"

code_dim=6 #10
hidden_dim=256 
cond_dim = 1
model = VelocityField(code_dim, hidden_dim, cond_dim)

for i in range(3):
    ckpt = torch.load(ckpt_path(experiment_name, model_names[0]))

    state_dict = {}
    for k in list(ckpt["state_dict"].keys()):
        prefix, key = k.split(".", 1)
        if prefix == "vf":
            state_dict[key] = ckpt["state_dict"][f"{prefix}.{key}"]

    model.load_state_dict(
        state_dict
    )
    model.push_to_hub(f"{HF_USERNAME}/spender-II-vf-{i}")

# could possibly clean this up using config loaders.
