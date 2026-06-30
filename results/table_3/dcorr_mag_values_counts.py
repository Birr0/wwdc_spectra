import os 
import argparse
from joblib import Parallel, delayed

import numpy as np
from datasets import load_dataset, load_from_disk
#import dcor
#from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import pandas as pd

load_dotenv() 

DATA_ROOT = os.getenv("DATA_ROOT")
z_lims=["0.100", "0.105", "0.110", "0.115", "0.120", "0.125", "0.130", "0.135", "0.140", "0.145", "0.150"]
z_lim_idx = {
    z_lim: idx for idx, z_lim in enumerate(z_lims)
}
embed_types = ["orig", "cond", "uncond"] 
SEED = 42

splits = ['test']
z_grid = np.linspace(0.001, 0.3, 300)
Z_TOL = 1e-4

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z_lim", type=str, default="0.050", help="Redshift cut as a str.")
    p.add_argument("--model_no", type=str, required=True, help="Model number identifier.")
    args = p.parse_args()
    
    z_lim = args.z_lim
    model_no = args.model_no

    fp = f"{DATA_ROOT}/sdss/magnitude_limited_embeddings_{model_no}/z=0.150"

    ds = load_from_disk(
        fp
    )

    results = []

    for embed_type in embed_types:
        # --- Process Test Data ---
        s_test = np.array(ds["test"][embed_type])
        z_test = np.array(ds["test"]["z"])
        mask_ratio_test = np.array(ds["test"]["mask_ratio"])

        max_z = float(z_lim)

        test_filter = (mask_ratio_test != 1.) & (z_test <= max_z) 
        test_indices = np.where(test_filter)[0]
        
        s_test = s_test[test_indices]
        z_test = z_test[test_indices]
        
        # Append counts to results
        results.append({
            "model_no": model_no,
            "z_lim": z_lim,
            "embed_type": embed_type,
            "s_test_len": len(s_test),
            "z_test_len": len(z_test)
        })
    
    # Convert results to a pandas DataFrame
    df_results = pd.DataFrame(results)
    
    # Ensure the output directory exists (as a backup to the bash script)
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV with unique filename per SLURM task
    csv_filename = os.path.join(output_dir, f"counts_model_{model_no}_z{z_lim}.csv")
    df_results.to_csv(csv_filename, index=False)
    
    print(f"Saved counts to {csv_filename}")
    
if __name__ == "__main__":
    main()