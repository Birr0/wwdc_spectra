import os 

from datasets import load_dataset
from dotenv import load_dotenv 
from huggingface_hub import login

load_dotenv()
login()

DATA_ROOT = os.getenv(
    "DATA_ROOT"
)
HF_USERNAME = os.getenv(
    "HF_USERNAME"
)

DATA_NAME = "sdss_II"
EXPERIMENT_NAME ="spender_II_flow"  #"spender_I_flow"
'''
models = [
    "6997867_0",  "6997867_1",  "6997867_2"
]
'''
models = [
    "7000609_0",  "7000609_1",  "7000609_2"
]

split = "test"
embedding_dirs = [
    f"{DATA_ROOT}/{DATA_NAME}/{EXPERIMENT_NAME}/embeddings/{model}/{split}" 
    for model in models
]

for i, embed_dir in enumerate(embedding_dirs):
    data_files = {"test": f"{embed_dir}/*.parquet"}  # label it correctly
    ds_dict = load_dataset("parquet", data_files=data_files)

    ds_dict.push_to_hub(
        f"{HF_USERNAME}/spender-II-vf-{i}",
        private=False,          # optional
    )
    print(embed_dir)