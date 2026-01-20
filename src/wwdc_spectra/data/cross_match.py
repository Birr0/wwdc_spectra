import os 

from astropy.table import Table 
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset 

load_dotenv()

EXT_CATALOG_COLUMNS = [
    "ObjID", "GLXID", "plate", "MJD", "fiberID", 
    "ra", "dec", "z", "rchi2", "logM*", "logM*_err", 
    "logSFR", "logSFR_err", "A_fuv", "A_fuv_err", 
    "A_b", "A_b_err", "A_v", "A_v_err", "flag_sed", 
    "UV_survey", "flag_uv", "flag_midir", "flag_mgs"
]

EXT_CATALOG_FLOAT_FEATURES = [
    "z", "rchi2", "logM*", "logM*_err", 
    "logSFR", "logSFR_err", "A_fuv", "A_fuv_err", 
    "A_b", "A_b_err", "A_v", "A_v_err"
]

EXT_CATALOG_INT_FEATURES = [
    "flag_sed", "UV_survey", "flag_uv", "flag_midir", "flag_mgs"
]

DATA_ROOT = os.getenv("DATA_ROOT")
CATALOG_DIR = DATA_ROOT + "/sdss/GSWLC-X2.dat"
MATCHED_CAT_FP = DATA_ROOT + "/sdss/GSWLC-X2-matched-test.csv"
HF_USERNAME = os.getenv("HF_USERNAME")

def _norm_id(x):
    if x is None:
        return None
    # decode bytes from HDF5
    if isinstance(x, (bytes, np.bytes_)):
        s = x.decode("ascii", "ignore").strip()
    else:
        s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    # canonicalize numeric-looking ids: "123.0" -> "123"
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s

def process_gswlc_catalog(
    catalog_dir=CATALOG_DIR
):
    catalog = Table.read(
        catalog_dir, 
        format="ascii.basic", 
        comment="#", 
        guess=False,
        names=EXT_CATALOG_COLUMNS
    ).to_pandas()

    catalog["ObjID"] = catalog["ObjID"].map(_norm_id)
    _ext_float_cols = list(EXT_CATALOG_FLOAT_FEATURES)
    _ext_int_cols   = list(EXT_CATALOG_INT_FEATURES)
    _keep = _ext_float_cols + _ext_int_cols

    return catalog, _keep


def process_embeddings(
    name="spender-I-vf-0"
):
    # Open embedding data.
    embedding_data = load_dataset(f"{HF_USERNAME}/{name}")["test"]
    # change harcoded test split later

    df = pd.DataFrame({
        "row": np.arange(len(embedding_data["id"])),
        "ObjID": [_norm_id(x) for x in embedding_data["id"]],
        "z_batch": embedding_data["z"]
    })
    return df


def match_catalog_to_embeddings(
    embeddings,
    catalog,
    keep_cols
):
    # state columns to keep
    keep_cols = ["ObjID", "z"] + [c for c in keep_cols if c != "z"]
    cat_small = catalog[keep_cols].copy()

    # 3) Merge on ObjID (duplicates in catalog are fine — this expands rows)
    m = embeddings.merge(cat_small, on="ObjID", how="left")

    # 4) Distance in z; NaNs -> +inf
    m["abs_dz"] = np.abs(m["z"] - m["z_batch"])
    m.loc[m["abs_dz"].isna(), "abs_dz"] = np.inf

    # 5) Pick the best match per batch row
    best_idx = m.groupby("row")["abs_dz"].idxmin()
    picked = m.loc[best_idx].sort_values("row").reset_index(drop=True)

    # 6) Enforce a tolerance (keep NaNs to preserve 1:1 alignment with batch)
    tol = 1e-4
    mask = picked["abs_dz"].to_numpy() <= tol
    picked.loc[~mask, keep_cols[1:]] = np.nan   # blank unmatched (z and features)

    # Inspect

    df = picked[['ObjID', 'z_batch', 'rchi2', 'logM*', 'logM*_err', 'logSFR',
        'logSFR_err', 'A_fuv', 'A_fuv_err', 'A_b', 'A_b_err', 'A_v', 'A_v_err',
        'flag_sed', 'UV_survey', 'flag_uv', 'flag_midir', 'flag_mgs']]

    df = df.rename(columns={
        "ObjID": "id",
        "z_batch": "z"
    })
    df["id"] = df["id"].astype(str)

    return df

def merge_matched_cat_with_embeddings(
    matched_cat="spectra_catalog",
    embeddings="spender-I-vf-0",
    key="id",
    join="left outer"
):
    # could add in local file checks.
    matched_cat = load_dataset(
        f"{HF_USERNAME}/{matched_cat}"
    )["test"].to_pandas()
    embeddings = load_dataset(
        f"{HF_USERNAME}/{embeddings}"
    )["test"].to_pandas()

    # remove hardcoded test split selection.
    # for future implementations.

    matched_df = pd.merge(
        matched_cat, 
        embeddings,
        'left'
    )
    
    return matched_df 

if __name__ == "__main__":
    pass 
    '''
    catalog, keep_cols = process_gswlc_catalog(
        catalog_dir=CATALOG_DIR
    )
    embeddings = process_embeddings(
        name="spender-I-vf-0"
    )

    matched_cat = match_catalog_to_embeddings(
        embeddings=embeddings,
        catalog=catalog,
        keep_cols=keep_cols
    )

    # save csv of the matched catalog.
    matched_cat.to_csv(
        MATCHED_CAT_FP
    )

    # upload to hugging face
    data_files = {"test": MATCHED_CAT_FP}  # label it correctly
    ds_dict = load_dataset("csv", data_files=data_files)
    # from datasets import Value, Features
    #new_features = ds_dict["test"].features.copy()
    #new_features["id"] = Value("string")
    #ds_dict = ds_dict.cast(Features(new_features))

    ds_dict.push_to_hub(
        f"{HF_USERNAME}/spectra_catalog",
        private=False,  # optional
    )
    '''

    #df = merge_matched_cat_with_embeddings()