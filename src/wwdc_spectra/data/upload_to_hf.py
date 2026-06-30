import os
import glob

from natsort import natsorted
from datasets import load_dataset 
from dotenv import load_dotenv

load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_ROOT = os.getenv("DATA_ROOT")

models = {
    "spender_I_flow": "7655991_0",
    "spender_II_flow": "7655992_0",
}

for model, model_no in models.items():
    fp = f"{DATA_ROOT}/sdss_II/{model}_v2/embeddings/{model_no}"
    train_files = natsorted(glob.glob(f"{fp}/train/*.parquet"))
    test_files = natsorted(glob.glob(f"{fp}/test/*.parquet"))
    val_files = natsorted(glob.glob(f"{fp}/val/*.parquet"))

    data_files = {
        "train": train_files,
        "test": test_files,
        "val": val_files
    }

    ds = load_dataset(
        "parquet", 
        data_files=data_files
    )

    print(ds)

    dataset_name = f"{model}_embeddings" 
    repo_id = f"{HF_USERNAME}/{dataset_name}"

    print(f"Uploading dataset to {repo_id}...")
    ds.push_to_hub(
        repo_id=repo_id,
        token=HF_TOKEN,
        private=False 
    )

    print("Upload complete!")