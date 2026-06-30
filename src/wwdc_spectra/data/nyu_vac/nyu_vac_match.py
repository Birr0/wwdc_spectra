import os 
import ast

from astropy.io import fits
from astropy.table import Table, hstack, join
from dotenv import load_dotenv
import numpy as np 
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from astropy.coordinates import SkyCoord
from astropy import units as u

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_ROOT = os.getenv("DATA_ROOT")

catalog_fp = DATA_ROOT + "/sdss/specObj-dr17.fits"
vac_specto_fp = DATA_ROOT + "/sdss/nyu_vac/object_sdss_spectro.fits"
vac_imaging_fp = DATA_ROOT + "/sdss/nyu_vac/object_sdss_imaging.fits"
main_galaxy_sample_catalog_fp = DATA_ROOT + "/sdss/dr17_main_galaxy_sample.fits"
kcorrect_fp = DATA_ROOT + "/sdss/nyu_vac/kcorrect.nearest.petro.z0.00.fits"
index_fp = DATA_ROOT + "/sdss/nyu_vac/dr17_main_galaxy_sample_vac_index.fits"
pos_fp = DATA_ROOT + "/sdss/nyu_vac/object_catalog.fits"

def load_fits(fp):
    return fits.open(fp, memmap=True)

def make_table(fits_data):
    return Table(fits_data)

def selection_fn(catalog):
    mask = catalog['SPECPRIMARY'] == 1            # Only use the primary spectrum for each object  
    mask &= catalog['TARGETTYPE'] == "SCIENCE "   # Only use science targets (ignore sky and others)
    mask &= catalog['PLATEQUALITY'] == "good    " # Only use plates with good status
    mask &= catalog["CLASS"] == "GALAXY"
    return mask

def make_main_galaxy_sample():
    dr17 = load_fits(catalog_fp)
    dr17_table = make_table(dr17[1].data)
    mask = selection_fn(dr17_table)
    _mask = np.array(mask)
    dr17_main_galaxy_sample = dr17_table[mask]
    dr17_main_galaxy_sample.write(
        DATA_ROOT + "/sdss/dr17_main_galaxy_sample.fits",
        format="fits",
        overwrite=True
    )
    return 

def load_raw_spectra(split="test"):
    DATA_DIR = os.getenv("DATA_ROOT") + "/sdss/sdss_II_catalog"
    dataset = load_from_disk(DATA_DIR)[split]
    return dataset

def load_dr17_main_sample():
    fp = DATA_ROOT + "/sdss/dr17_main_galaxy_sample.fits"
    ms = make_table(load_fits(fp)[1].data)
    return ms

def make_VAC_index():
    # 1. Load NYU VAGC (The DR7-based catalog)
    pos_fits = load_fits(pos_fp)
    
    # Ensure RA/DEC columns are present (usually 'RA' and 'DEC' or 'ra', 'dec')
    pos_table = Table(pos_fits[1].data)
    pos_table["VAC_ID"] = np.arange(len(pos_table))
    
    dataset = {}
    for split in ["test", "val", "train"]:
        print(f"Starting {split} split")
        data = {}
        # 2. Load DR17 (Your target catalog)
        dr17_table = load_raw_spectra(split) #load_dr17_main_sample() # Make sure this has RA/DEC

        # 3. Create SkyCoord objects
        # Note: Check if your VAGC columns are uppercase or lowercase
        vac_coords = SkyCoord(ra=pos_table['RA']*u.degree, dec=pos_table['DEC']*u.degree)
        dr17_coords = SkyCoord(ra=dr17_table['ra']*u.degree, dec=dr17_table['dec']*u.degree)

        # 4. Perform the Cross-Match
        # idx are the indices into dr17_table that match each row of vac_table
        idx, d2d, _ = vac_coords.match_to_catalog_sky(dr17_coords)

        # 5. Define a matching threshold (1 or 2 arcseconds is standard for SDSS)
        max_sep = 1.0 * u.arcsec
        mask = d2d < max_sep
        
        # 6. Build the index table
        # We only take the rows from VAGC that found a match
        matched_vac = pos_table[mask]        
        matched_dr17 = dr17_table[idx[mask]]
        # Create the mapping table
        data['VAC_ID'] = np.array(matched_vac['VAC_ID'])
        data["object_id"] = np.array(matched_dr17["object_id"])
        data['separation_arcsec'] = np.array(d2d[mask].to(u.arcsec).value)
        data['PLUG_RA'] = np.array(matched_dr17['ra'])
        data['PLUG_DEC'] = np.array(matched_dr17['dec'])
        dataset[split] = data
        print(f"Matched {len(data['VAC_ID'])} galaxies out of {len(pos_table)} VAGC entries and {len(dr17_table)}")
        print(f"Finsihed {split} split")
        #break

    ds_dict = DatasetDict({
        split: Dataset.from_dict(data)
        for split, data in dataset.items()
    })
    ds_dict.push_to_hub(
        f"{HF_USERNAME}/VACG_raw_cross_match",
        private=False, 
    )
    
    return #result

def make_VAC_index():
    # 1. Load NYU VAGC (The DR7-based catalog)
    pos_fits = load_fits(pos_fp)
    pos_table = Table(pos_fits[1].data)
    pos_table["VAC_ID"] = np.arange(len(pos_table))

    dataset = {}
    for split in ["test", "val", "train"]:
        print(f"Starting {split} split")

        # load dr17 for this split
        dr17_table = load_raw_spectra(split)

        # build SkyCoords & match as before (you already make arrays)
        vac_coords = SkyCoord(ra=pos_table['RA']*u.degree, dec=pos_table['DEC']*u.degree)
        dr17_coords = SkyCoord(ra=dr17_table['ra']*u.degree, dec=dr17_table['dec']*u.degree)

        idx, d2d, _ = vac_coords.match_to_catalog_sky(dr17_coords)
        max_sep = 1.0 * u.arcsec
        mask = d2d < max_sep

        matched_vac = pos_table[mask]
        matched_dr17 = dr17_table.select(idx[mask])
        # create numpy arrays (you already do this; kept here for clarity)
        data = {
            'VAC_ID': np.array(matched_vac['VAC_ID'], dtype=np.int64),
            #'object_id': np.array(matched_dr17['object_id']),
            'separation_arcsec': np.array(d2d[mask].to(u.arcsec).value, dtype=np.float32),
            'PLUG_RA': np.array(matched_dr17['ra'], dtype=np.float64),
            'PLUG_DEC': np.array(matched_dr17['dec'], dtype=np.float64),
        }
        # after matched_dr17 = ...
        obj_raw = np.array(
            [ast.literal_eval(x).decode('utf-8').strip() for x in matched_dr17['object_id']]
        )
        # use obj_int in your dict
        data['object_id'] = obj_raw
        dataset[split] = data
        print(f"Matched {len(data['VAC_ID'])} galaxies out of {len(pos_table)} VAGC entries and {len(dr17_table)}")
        print(f"Finished {split} split")

    # --- Build datasets from pyarrow tables (fast, low-copy) ---
    ds_dict = DatasetDict({
        split: Dataset.from_dict(data)
        for split, data in dataset.items()
    })
    # Save locally to inspect and avoid repeated failed uploads
    local_path = f"{DATA_ROOT}/sdss/VACG_raw_cross_match_dataset"
    ds_dict.save_to_disk(local_path)
    print(f"Saved dataset to {local_path} (inspect before pushing if you'd like)")
    ds_dict.push_to_hub(f"{HF_USERNAME}/VACG_raw_cross_match", private=False)
    return #ds_dict

def make_value_added_catalog_sample(
    catalog_fp,
    sample_fp
):
    # Join on the VAC index for each table in the VAC.
    idx_fits = load_fits(
        DATA_ROOT + "/sdss/nyu_vac/dr17_main_galaxy_sample_vac_index_pos_match.fits" #DATA_ROOT + "/sdss/nyu_vac/dr17_main_galaxy_sample_vac_index.fits"
    )
    idx_table = make_table(idx_fits[1].data)
    cata_fits = load_fits(
        catalog_fp
    )
    cata_table = make_table(cata_fits[1].data)
    cata_table["VAC_ID"] = np.arange(len(cata_table))
    # you cant
    merged_table = join(
        idx_table,
        cata_table,
        keys=["VAC_ID"],
        join_type="inner"
    )
    merged_table.write(
        DATA_ROOT + f"/sdss/nyu_vac/galaxy_sample/{sample_fp}",
        format="fits",
        overwrite=True
    )
    print(f"Matched {len(merged_table)} galaxies out of {len(idx_table)} VAGC entries.")
    return

def make_image_spectro_catalog_sample():
    # Join on the VAC index for each table in the VAC.
    idx_fits = load_fits(
        DATA_ROOT + "/sdss/nyu_vac/dr17_main_galaxy_sample_vac_index_pos_match.fits" #DATA_ROOT + "/sdss/nyu_vac/dr17_main_galaxy_sample_vac_index.fits"
    )
    idx_table = make_table(idx_fits[1].data)
    imaging_fits = load_fits(
        vac_imaging_fp
    )
    spectro_fits = load_fits(
        vac_specto_fp
    )

    spectro_table = make_table(imaging_fits[1].data)
    imaging_table = make_table(spectro_fits[1].data)

    joint_table = hstack([spectro_table, imaging_table])
    joint_table["VAC_ID"] = np.arange(len(joint_table))

    merged_table = join(
        idx_table,
        joint_table,
        keys=["VAC_ID"],
        join_type="inner",
        table_names=["_idx_tab", ""]
    )
    merged_table.write(
        DATA_ROOT + f"/sdss/nyu_vac/galaxy_sample/image_spectro.fits",
        format="fits",
        overwrite=True
    )
    print(f"Matched {len(merged_table)} galaxies out of {len(idx_table)} VAGC entries. ")
    return
    
def to_native(arr, swap=True):
    arr = np.asarray(arr)
    if swap:
        # Convert non-native endian to native
        if arr.dtype.byteorder not in ('=', '|'):
            arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
    return arr

def upload_k_correct_VAC_sample_to_HF():
    sample_fp = "k_correct_main_galaxy_sample_pos_match.fits" #"k_correct_main_galaxy_sample.fits"
    fp = DATA_ROOT + f"/sdss/nyu_vac/galaxy_sample/{sample_fp}"
    kcorr = load_fits(fp)
    kcorr_table = make_table(kcorr[1].data)

    print(len(kcorr_table))
    # get columns that have multiple entires and cast as lists.
    data = {}
    sdss_filters = ["u", "g", "r", "i", "z"]
    filters = sdss_filters + ["J", "H", "K"]

    single_cols = [
        "VAC_ID", "RA", "DEC", "Z", "MASS", "INTSFH", "METS", "B300", "B1000", "PLUG_RA", "PLUG_DEC"
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
        f"{HF_USERNAME}/kcorrect_VAC_galaxy_sample",
        private=False, 
    )
    return 


def upload_spectro_image_sample_to_HF():
    sample_fp = "image_spectro.fits"
    fp = DATA_ROOT + f"/sdss/nyu_vac/galaxy_sample/{sample_fp}"
    image_spectro_fits = load_fits(fp)
    image_spectro_table = make_table(image_spectro_fits[1].data)
    #image_spectro_table["PLUG_RA"] = image_spectro_table["PLUG_"]
    # get columns that have multiple entires and cast as lists.
    data = {}
    filters = ["u", "g", "r", "i", "z"]

    single_cols = [
        'SDSS_IMAGING_TAG','SDSS_IMAGING_TAG_PRIMARY','RUN','RERUN','CAMCOL','FIELD','ID','RA','DEC',
        'NCHILD','RESOLVE_STATUS','FLAGS','FLAGS2','OBJC_ROWC','OBJC_COLC','COLC','OBJC_FLAGS','OBJC_FLAGS2',
        'PETROR50','PETROR90', 'SKYFLUX', 'EXTINCTION','NMGYPERCOUNT','VAGC_SELECT','PSP_SKYFLUX','CALIBOBJ_POSITION',
        'SCORE','CALIB_STATUS','IFIELD','BALKAN_ID', 'FRACPSF','DEVFLUX','DEVFLUX_IVAR','EXPFLUX','EXPFLUX_IVAR',
        'SDSS_SPECTRO_TAG', 'SDSS_SPECTRO_TAG_PRIMARY','SDSS_SPECTRO_INGROUP','SDSS_SPECTRO_FIRSTGROUP','SDSS_SPECTRO_NEXTGROUP',
        'SDSS_SPECTRO_MULTGROUP','PROGNAME','CHUNKNAME','PLATEQUALITY','PLATESN2','PRIMTARGET',
        'SECTARGET','SPECPRIMARY','SPECOBJ_ID','NSPECOBS','CALIBFLUX','CALIBFLUX_IVAR','PLATE',
        'TILE','MJD','FIBERID','OBJID','OBJTYPE','PLUG_RA','PLUG_DEC','CLASS','SUBCLASS',
        'Z','Z_ERR','RCHI2','DOF','RCHI2DIFF','TFILE','TCOLUMN','NPOLY','THETA','THETA_COVAR',
        'VDISP','VDISP_ERR','VDISPZ','VDISPZ_ERR','VDISPCHI2','VDISPNPIX','VDISPDOF','WAVEMIN','WAVEMAX',
        'WCOVERAGE','ZWARNING','SN_MEDIAN','CHI68P','FRACNSIGMA','FRACNSIGHI','FRACNSIGLO', 'SPECTROSKYFLUX',
        'ANYANDMASK','ANYORMASK','SPEC1_G','SPEC1_R','SPEC1_I', 'SPEC2_G','SPEC2_R','SPEC2_I','VAC_ID','BESTOBJID'
    ]
    filter_cols = [
        'MODELFLUX','MODELFLUX_IVAR','PETROFLUX','PETROFLUX_IVAR','FIBERFLUX','PSFFLUX','PSFFLUX_IVAR',
        'SPECTROFLUX','SPECTROFLUX_IVAR', 'SPECTROSYNFLUX','SPECTROSYNFLUX_IVAR'
    ]

    for col in single_cols:
        data[f"{col}"] = to_native(
            image_spectro_table[f"{col}"]
        )

    for col in filter_cols:
        for idx, filter_ in enumerate(filters):
            data[f"{col}_{filter_}"] = to_native(
                image_spectro_table[f"{col}"][:, idx]
            )

    ds = Dataset.from_dict(data)
    ds_dict = DatasetDict({
        "full": ds
    })

    ds_dict.push_to_hub(
        f"{HF_USERNAME}/spectro_image_VAC_galaxy_sample",
        private=False, 
    )
    return

if __name__ == "__main__":
    #make_main_galaxy_sample()
    #load_dr17_main_sample()
    print("Starting")
    #make_VAC_index()
    make_value_added_catalog_sample(
        kcorrect_fp,
        "k_correct_main_galaxy_sample_pos_match.fits"
    )
    #upload_k_correct_VAC_sample_to_HF()

    #make_image_spectro_catalog_sample()
    #upload_spectro_image_sample_to_HF()

'''
from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd

ds_cata = load_dataset("Birr001/spectra_catalog")
ds = load_dataset("Birr001/spectro_image_VAC_galaxy_sample")["full"]

# 1. Load NYU VAGC (The DR7-based catalog)
pos_fits = load_fits(pos_fp)
# Ensure RA/DEC columns are present (usually 'RA' and 'DEC' or 'ra', 'dec')
pos_table = Table(pos_fits[1].data)
pos_table["VAC_ID"] = np.arange(len(pos_table))

# 2. Load DR17 (Your target catalog)
raw_table = load_raw_spectra()
# 3. Create SkyCoord objects
# Note: Check if your VAGC columns are uppercase or lowercase
vac_coords = SkyCoord(ra=pos_table['RA']*u.degree, dec=pos_table['DEC']*u.degree)
raw_coords = SkyCoord(ra=raw_table['ra']*u.degree, dec=raw_table['dec']*u.degree)

# 4. Perform the Cross-Match
# idx are the indices into dr17_table that match each row of vac_table
idx, d2d, _ = vac_coords.match_to_catalog_sky(raw_coords)

# 5. Define a matching threshold (1 or 2 arcseconds is standard for SDSS)
max_sep = 1.0 * u.arcsec
mask = d2d < max_sep

# 6. Build the index table
# We only take the rows from VAGC that found a match
matched_vac = pos_table[mask]
matched_dr17 = raw_table[idx[mask]]

# Create the mapping table
result = Table()
result['VAC_ID'] = matched_vac['VAC_ID']
result['BESTOBJID'] = matched_dr17['BESTOBJID']
result['separation_arcsec'] = d2d[mask].to(u.arcsec)

print(f"Matched {len(result)} galaxies out of {len(pos_table)} VAGC entries.")
'''

cols = ['SDSS_IMAGING_TAG','SDSS_IMAGING_TAG_PRIMARY','RUN','RERUN','CAMCOL','FIELD','ID','RA','DEC',
    'NCHILD','RESOLVE_STATUS','FLAGS','FLAGS2','OBJC_ROWC','OBJC_COLC','COLC','OBJC_FLAGS','OBJC_FLAGS2',
    'SKYFLUX','MODELFLUX','MODELFLUX_IVAR','PETROFLUX','PETROFLUX_IVAR','FIBERFLUX','PSFFLUX','PSFFLUX_IVAR','PETROR50','PETROR90',
    'EXTINCTION','NMGYPERCOUNT','VAGC_SELECT','PSP_SKYFLUX','CALIBOBJ_POSITION','SCORE','CALIB_STATUS','IFIELD','BALKAN_ID',
    'FRACPSF','DEVFLUX','DEVFLUX_IVAR','EXPFLUX','EXPFLUX_IVAR','SDSS_SPECTRO_TAG',
    'SDSS_SPECTRO_TAG_PRIMARY','SDSS_SPECTRO_INGROUP','SDSS_SPECTRO_FIRSTGROUP','SDSS_SPECTRO_NEXTGROUP',
    'SDSS_SPECTRO_MULTGROUP','PROGNAME','CHUNKNAME','PLATEQUALITY','PLATESN2','PRIMTARGET',
    'SECTARGET','SPECPRIMARY','SPECOBJ_ID','NSPECOBS','CALIBFLUX','CALIBFLUX_IVAR','PLATE',
    'TILE','MJD','FIBERID','OBJID','OBJTYPE','PLUG_RA','PLUG_DEC','CLASS','SUBCLASS',
    'Z','Z_ERR','RCHI2','DOF','RCHI2DIFF','TFILE','TCOLUMN','NPOLY','THETA','THETA_COVAR',
    'VDISP','VDISP_ERR','VDISPZ','VDISPZ_ERR','VDISPCHI2','VDISPNPIX','VDISPDOF','WAVEMIN','WAVEMAX',
    'WCOVERAGE','ZWARNING','SN_MEDIAN','CHI68P','FRACNSIGMA','FRACNSIGHI','FRACNSIGLO','SPECTROFLUX','SPECTROFLUX_IVAR',
    'SPECTROSYNFLUX','SPECTROSYNFLUX_IVAR','SPECTROSKYFLUX','ANYANDMASK','ANYORMASK','SPEC1_G','SPEC1_R','SPEC1_I',
    'SPEC2_G','SPEC2_R','SPEC2_I','VAC_ID','BESTOBJID'
]
