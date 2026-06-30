import os
import joblib
import argparse

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
import pandas as pd
from datasets import load_dataset

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME")
DATA_ROOT = os.getenv("DATA_ROOT")

embed_types = ["orig", "cond", "uncond"]

# Initialize storage
results_list = []
plot_data = {}  # To store the actual arrays for plotting

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, default=0, help="Re-run number of the embedding using different flow model initialisation.")
    p.add_argument("--embed_type", type=str, default="orig", help="Embed type: orig, cond or uncond")
    p.add_argument("--base_model", type=str, default="spender-I", help="Base model: spender I/II")

    args = p.parse_args()
    idx = args.idx
    embed_type = args.embed_type
    base_model = args.base_model
    
    print(idx, embed_type)
    
    if base_model == "spender_I":
        fp = f"{DATA_ROOT}/sdss_II/spender_I_flow_v2/embeddings/7655991_{idx}"
    elif  base_model == "spender_II":
        fp = f"{DATA_ROOT}/sdss_II/spender_II_flow_v2/embeddings/7655992_{idx}"
    else:
        print("Base model not accepted.")
    
    print(fp)
    data_files = {
        "train": f"{fp}/train/*.parquet",
        "test": f"{fp}/test/*.parquet",
        # "val": f"{fp}/val/*.parquet",
    }
    ds = load_dataset(
        "parquet",
        data_files=data_files
    )

    # 1. Convert to numpy efficiently
    s_train = np.array(ds["train"][embed_type])
    z_train = np.array(ds["train"]["z"])

    mask_ratio_train = np.array(ds["train"]["mask_ratio"])
    s_train = s_train[mask_ratio_train != 1.]
    z_train = z_train[mask_ratio_train != 1.]

    s_test = np.array(ds["test"][embed_type])
    z_test = np.array(ds["test"]["z"])

    mask_ratio_test = np.array(ds["test"]["mask_ratio"])
    s_test = s_test[mask_ratio_test != 1.]
    z_test = z_test[mask_ratio_test != 1.]

    # 2. Fit and Predict with a simple MLP
    model = MLPRegressor(
        hidden_layer_sizes=(64,64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        batch_size=256, 
    )

    model.fit(s_train, z_train)
    z_pred = model.predict(s_test)

    # 3. Calculate Metric
    r2 = r2_score(z_test, z_pred)

    # 4. Store metrics for a summary table
    results_list.append({
        "dataset_index": idx,
        "embed_type": embed_type,
        "r2": r2
    })

    os.makedirs(f"./mlp_models_{base_model}_v2", exist_ok=True)
    os.makedirs(f"./mlp_predictions_{base_model}_v2", exist_ok=True)

    joblib.dump(model, f"./mlp_models_{base_model}_v2/{embed_type}_{idx}.joblib")

    # Create summary DataFrame
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(f"./mlp_predictions_{base_model}_v2/{embed_type}_{idx}.csv", index=False)