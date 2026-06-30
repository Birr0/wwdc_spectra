import os 

import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack, join
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")
HF_USERNAME = os.getenv("HF_USERNAME")

sersic_fp = DATA_ROOT + "/sdss/nyu_vac/sersic_catalog.fits" #sersicect_none_model_z_010.fits"

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

sersic = load_fits(sersic_fp)
sersic_table = make_table(sersic[1].data)
sersic_table["VAC_ID"] = np.arange(len(sersic_table))

# get columns that have multiple entires and cast as lists.

data = {}
filters = ["u", "g", "r", "i", "z"]

single_cols = [
    "VAC_ID"
]
filter_cols = [
    "SERSIC_AMP", "SERSIC_R0", "SERSIC_N", "SERSIC_CHI2", "SERSIC_NPROF",
    "SERSIC_FLUX", "SERSIC_R50", "SERSIC_R90", "SERSIC_PETROFLUX", "SERSIC_PETROR50",
    "SERSIC_PETROR90", "SERSIC_CANON_AMP", "SERSIC_CANONFLUX", "SERSIC_CANONPETROFLUX"
]

# "SERSIC_COVAR"

for col in single_cols:
    data[f"{col}"] = to_native(
        sersic_table[f"{col}"]
    )

for col in filter_cols:
    for idx, filter_ in enumerate(filters):
        data[f"{col}_{filter_}"] = to_native(
            sersic_table[f"{col}"][:, idx]
        )

ds = Dataset.from_dict(data)
ds_dict = DatasetDict({
    "full": ds
})

ds_dict.push_to_hub(
    f"{HF_USERNAME}/sersic_VAC",
    private=False, 
)