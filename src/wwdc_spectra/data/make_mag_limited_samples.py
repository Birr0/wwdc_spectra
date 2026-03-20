import os 
import ast
import argparse

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from datasets import load_from_disk, Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv 
from astropy.cosmology.realizations import Planck18
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

cosmo = Planck18

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME")
DATA_ROOT = os.getenv("DATA_ROOT")

splits = ['test', 'train', 'val']
z_grid = np.linspace(0.001, 0.3, 300)

cols = [
    'VAC_ID',
    'RA',
    'DEC',
    'Z_kcorr',
    'MASS',
    'INTSFH',
    'METS',
    'B300',
    'B1000',
    'ABMAGGIES_u',
    'ABMAGGIES_g',
    'ABMAGGIES_r',
    'ABMAGGIES_i',
    'ABMAGGIES_z',
    'ABMAGGIES_J',
    'ABMAGGIES_H',
    'ABMAGGIES_K',
    'ABMAGGIES_IVAR_u',
    'ABMAGGIES_IVAR_g',
    'ABMAGGIES_IVAR_r',
    'ABMAGGIES_IVAR_i',
    'ABMAGGIES_IVAR_z',
    'ABMAGGIES_IVAR_J',
    'ABMAGGIES_IVAR_H',
    'ABMAGGIES_IVAR_K',
    'KCORRECT_u',
    'KCORRECT_g',
    'KCORRECT_r',
    'KCORRECT_i',
    'KCORRECT_z',
    'KCORRECT_J',
    'KCORRECT_H',
    'KCORRECT_K',
    'ABSMAG_u',
    'ABSMAG_g',
    'ABSMAG_r',
    'ABSMAG_i',
    'ABSMAG_z',
    'ABSMAG_J',
    'ABSMAG_H',
    'ABSMAG_K',
    'ABSMAG_IVAR_u',
    'ABSMAG_IVAR_g',
    'ABSMAG_IVAR_r',
    'ABSMAG_IVAR_i',
    'ABSMAG_IVAR_z',
    'ABSMAG_IVAR_J',
    'ABSMAG_IVAR_H',
    'ABSMAG_IVAR_K',
    'MTOL_u',
    'MTOL_g',
    'MTOL_r',
    'MTOL_i',
    'MTOL_z',
    'MTOL_J',
    'MTOL_H',
    'MTOL_K',
    'spectrum',
    'ra',
    'dec',
    'BESTOBJID',
    'VDISP',
    'VDISP_ERR',
    'Z_raw',
    'Z_ERR',
    'ZWARNING',
    'CLASS',
    'SOURCETYPE',
    'object_id',
    'separation_arcsec',
    'PLUG_RA',
    'PLUG_DEC'
]

def load_fits(fp):
    return fits.open(fp, memmap=True)

def make_table(fits_data):
    return Table(fits_data)

def load_raw_spectra(split="test"):
    DATA_DIR = os.getenv("DATA_ROOT") + "/sdss/sdss_II_catalog"
    dataset = load_from_disk(DATA_DIR)[split]
    return dataset

def to_native(arr, swap=True):
    arr = np.asarray(arr)
    if swap:
        # Convert non-native endian to native
        if arr.dtype.byteorder not in ('=', '|'):
            arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
    return arr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z_idx", type=int, default=49, help="Index of redshift.")
    args = p.parse_args()
    z_idx = args.z_idx
    z_max_idx = 149 # z = 0.15 

    local_path = f"{DATA_ROOT}/sdss/VACG_raw_cross_match_dataset"

    #ds = load_dataset("Birr001/") # VACG_cross_match")
    ds = load_from_disk(local_path)
    ds_kcorr = load_dataset("Birr001/kcorrect_VAC")
    df_kcorr = ds_kcorr["full"].to_pandas()

    cosmo = FlatLambdaCDM(H0=100, Om0=0.3) # Cosmology used in the SDSS sample
    D_L = cosmo.luminosity_distance(z_grid).to(u.pc)
    d0 = 10. * u.pc
    DM = 5 * np.log10(D_L/d0)
    m_lim = 17.77
    M_lim_curve = m_lim - DM

    jts = {}
    for split in splits:
        ds_raw = load_raw_spectra(split)
        raw_spectra = ds_raw.to_pandas()
        raw_spectra["object_id"] = raw_spectra["object_id"].apply(
            lambda x: ast.literal_eval(x).decode('utf-8').strip()
        )
        vac = ds[split].to_pandas()
        jts[split] = pd.merge(
            raw_spectra,
            vac,
            how="inner",
            on="object_id"
        )

    dataset = {}
    z_cut = z_grid[z_idx]
    for split in splits:
        merged_data = pd.merge(
            df_kcorr,
            jts[split],
            on="VAC_ID",
            suffixes=["_kcorr", "_raw"]
        )
        print(len(merged_data))
        #M_lim = M_lim_curve[z_idx]
        M_lim_zmax = M_lim_curve[z_max_idx]
        print(M_lim_zmax)

        M_r = np.array(merged_data["ABSMAG_r"])
        z = np.array(merged_data["Z_kcorr"])

        z_k = merged_data["Z_kcorr"].to_numpy()
        z_r = merged_data["Z_raw"].to_numpy()
        mask_z_agree = np.isclose(z_k, z_r, atol=1e-3, rtol=0)

        mask_box = (z > 0) & (z <= z_cut) & (M_r <= M_lim_zmax).reshape(-1) & mask_z_agree
        print(mask_box.sum(), len(M_r))
        indices = np.where(mask_box)[0]

        # Drop any rows that 'Z_kcorr', 'Z_raw' does not match to within 1e-3
        merged_data = merged_data.iloc[indices]
        # rename Z_raw to Z
        merged_data.rename(columns={"Z_raw": "Z"})
        # select columns from cols
        merged_data_filtered = merged_data[cols]
        dataset[split] = Dataset.from_dict({
            k: to_native(v) for k,v in merged_data_filtered.items()
        })
        print(f"Done split: {split}")

    ds_dict = DatasetDict(dataset)
    ds_dict.save_to_disk(f"{DATA_ROOT}/sdss/magnitude_limited/z={z_cut.item():.3f}")
    ds_dict.push_to_hub(
        f"{HF_USERNAME}/mag_limited_z_{z_cut.item():.3f}_spectra",
        private=False, 
    )
    print(f"Pushed dataset: {z_cut.item():.3f}")

if __name__ == "__main__":
    main()