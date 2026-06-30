import os 
import argparse

from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv() 

DATA_ROOT = os.getenv("DATA_ROOT")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z_lim", type=str, default="0.050", help="Redshift cut as a str.")
    args = p.parse_args()
    z_lim = args.z_lim
    print(z_lim)

    ds_name = f"Birr001/vol_limited_z_{z_lim}_spectra"
    print(ds_name)
    dataset = load_dataset(
        ds_name
    )
    dataset.save_to_disk(
        f"{DATA_ROOT}/sdss/volume_limited/z={z_lim}"
    )
    
if __name__ == "__main__":
    main()