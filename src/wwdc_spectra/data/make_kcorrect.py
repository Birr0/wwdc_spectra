import os 

import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack, join
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")
HF_USERNAME = os.getenv("HF_USERNAME")

kcorr_fp = DATA_ROOT + "/sdss/nyu_vac/kcorrect.nearest.petro.z0.00.fits" #kcorrect_none_model_z_010.fits"

def load_fits(fp):
    return fits.open(fp, memmap=True)

def make_table(fits_data):
    return Table(fits_data)

def to_native(arr, swap=True):
    arr = np.asarray(arr)
    if swap:
        # Convert non-native endian to native
        if arr.dtype.byteorder not in ('=', '|'):
            arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
    return arr

kcorr = load_fits(kcorr_fp)
kcorr_table = make_table(kcorr[1].data)
kcorr_table["VAC_ID"] = np.arange(len(kcorr_table))

# get columns that have multiple entires and cast as lists.

data = {}
sdss_filters = ["u", "g", "r", "i", "z"]
filters = sdss_filters + ["J", "H", "K"]

single_cols = [
    "VAC_ID", "RA", "DEC", "Z", "MASS", "INTSFH", "METS", "B300", "B1000"
]
filter_cols = [
    "ABMAGGIES", "ABMAGGIES_IVAR", "KCORRECT", 
    "ABSMAG", "ABSMAG_IVAR", "MTOL"
]

for col in single_cols:
    data[f"{col}"] = to_native(
        kcorr_table[f"{col}"]
    )

for col in filter_cols:
    for idx, filter_ in enumerate(filters):
        data[f"{col}_{filter_}"] = to_native(
            kcorr_table[f"{col}"][:, idx]
        )

for idx in range(5):
    data[f"COEFFS_{idx}"] = to_native(kcorr_table["COEFFS"][:, idx])


ds = Dataset.from_dict(data)
ds_dict = DatasetDict({
    "full": ds
})

ds_dict.push_to_hub(
    f"{HF_USERNAME}/kcorrect_VAC",
    private=False, 
)