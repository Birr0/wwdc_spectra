import os

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login()

DATA_ROOT = os.getenv("DATA_ROOT")
HF_USERNAME = os.getenv("HF_USERNAME")

DATA_NAME = "sdss_II"
EXPERIMENT_NAME = "spender_I_flow" #"spender_I_alpha_flow" #"spender_I_flow_volume_limited" #"spender_I_flow_single_step"  #"spender_II_flow"

'''models = [
    "6997867_0",  "6997867_1",  "6997867_2"
]'''

'''models = [
    f"7207317_{i}" for i in range(3) #"7206974_0" #"6997867_0" #"7112821_0" #"7000609_0",  "7000609_1",  "7000609_2"
]
'''
zs = [0.075, 0.100, 0.125]

models = [
    f"vol_limited/{z}"  for z in z #7112821_0" #"vol_limited6997867_0"
]

embedding_dirs = [
    f"{DATA_ROOT}/{DATA_NAME}/{EXPERIMENT_NAME}/embeddings/{model}"
    for model in models
]

 #, 0.075 0.150
model_no = "6997867_"
for i, embed_dir in enumerate(embedding_dirs):
    for j in range(3):
        data_files = {
            "train" : f"{embed_dir}/{model_no}_{j}/train/*.parquet",
            "test": f"{embed_dir}/{model_no}_{j}/test/*.parquet",
            "val": f"{embed_dir}/{model_no}_{j}/val/*.parquet"
        }  # label it correctly
        ds_dict = load_dataset("parquet", data_files=data_files)

        ds_dict.push_to_hub(
            f"{HF_USERNAME}/{EXPERIMENT_NAME}-z-{z[i]:.2f}-{model_no}_{j}-complete",
            private=False,          # optional
        )
        print(embed_dir)
