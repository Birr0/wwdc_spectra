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

'''
def single_bootstrap_test(seed, X, y, test_stat):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.integers(0, n, size=n)   # sample indices with replacement
    Xb = X[idx]
    yb = y[idx]

    return test_stat(Xb, yb)

def bootstrap_test(seed, X, y, B=500, test_stat=dcor.distance_covariance, n_jobs=1):
    seeds = np.arange(seed, seed + B)
    stats = Parallel(n_jobs=n_jobs)(
        delayed(single_bootstrap_test)(int(s), X, y, test_stat)
        for s in seeds
    )
    stats = np.array(stats)
    return stats
'''

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z_lim", type=str, default="0.050", help="Redshift cut as a str.")
    p.add_argument("--model_no", type=str, required=True, help="Model number identifier.")
    args = p.parse_args()
    
    z_lim = args.z_lim
    model_no = args.model_no

    results_list = []
    fp = f"{DATA_ROOT}/sdss/magnitude_limited_embeddings_{model_no}/z=0.150"

    ds = load_from_disk(
        fp
    )

    for embed_type in embed_types:
        # --- Process Test Data ---
        s_test = np.array(ds["test"][embed_type])
        z_test = np.array(ds["test"]["z"])
        mask_ratio_test = np.array(ds["test"]["mask_ratio"])

        max_z = float(z_lim)
        min_z = max_z - 0.005

        test_filter = (mask_ratio_test != 1.) & (z_test > min_z) & (z_test <= max_z) 
        test_indices = np.where(test_filter)[0]
        
        s_test = s_test[test_indices]
        z_test = z_test[test_indices]
        print(len(s_test), len(z_test))

        '''
        dist = dcor.distance_correlation(s_test, z_test)
        
        # NOTE: Changed n_jobs to 4 to utilize SLURM allocated CPUs
        result = bootstrap_test(
            SEED,
            s_test, 
            z_test,
            test_stat=dcor.distance_correlation,  
            n_jobs=4,
            B=1000
        )
        median_dist, lb_dist, ub_dist = np.percentile(result, [50.0, 2.5, 97.5])

        result_dict = {
            "dist": dist,
            "med_dist": median_dist,
            "dist_5": lb_dist,
            "dist_95": ub_dist,
            "samples": result,
            "z": z_test,
            "embed_type": embed_type,
            "data_size": test_filter.sum()
        }
        results_list.append(result_dict)
        print(10*"=")

        output_dir = "./mag_floor"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Pivot the list of dicts into a single dict of lists
        keys = results_list[0].keys()
        save_dict = {}
        
        for key in keys:
            extracted_list = [res[key] for res in results_list]
            
            try:
                save_dict[key] = np.array(extracted_list)
            except ValueError:
                save_dict[key] = np.array(extracted_list, dtype=object)

        # 3. Unpack the dictionary directly into the save function using **
        save_path = f"{output_dir}/{model_no}_{z_lim}.npz"
        np.savez_compressed(save_path, **save_dict)
        '''
    
if __name__ == "__main__":
    main()