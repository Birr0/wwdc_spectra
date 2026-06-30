import os
import glob

import numpy as np 
import pandas as pd 
import lightning as L
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from astroML.datasets import fetch_sdss_specgals
from natsort import natsorted
from dotenv import load_dotenv 

from wwdc_spectra.data.gswlc.cross_match import merge_matched_cat_with_embeddings

load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME")
DATA_ROOT = os.getenv("DATA_ROOT")

class EmbeddingDataset(Dataset):
    def __init__(
        self, 
        embeddings_fp,
        catalog_dset_name,
        split,
        embed_type, # ["orig", "cond",  "uncond"]
        params = ["z"] #, 'logM*', "logSFR", "A_v"]
    ):
        """
        Args:
            embedding_dir (str): The directory containing the embeddings.
        """
        
        dset = merge_matched_cat_with_embeddings(
            embeddings_fp=embeddings_fp,
            matched_cat=catalog_dset_name,
            split=split
        )

        dset = dset[dset[embed_type].notna()].reset_index(drop=True) # Drop NaN
        mask = ((dset[list(params)] != -99).all(axis=1)) & (dset["mask_ratio"] != 1.) # Drop -99 Filler values
        dset = dset[mask].reset_index(drop=True)

        self.X = dset[embed_type].to_numpy()          # shape: (N, D)
        self.Y = dset[list(params)].to_numpy(dtype=np.float32)  # shape: (N, P)
        self.mask_ratio = dset["mask_ratio"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
        mask_ratio = self.mask_ratio[idx]
        return x, y, mask_ratio


class JHU_MPA_EmbeddingDataset(Dataset):
    def __init__(
        self, 
        embeddings_fp,
        split,
        embed_type, # ["orig", "cond",  "uncond"]
        params = ["z"] #, 'logM*', "logSFR", "A_v"]
    ):
        """
        Args:
            embedding_dir (str): The directory containing the embeddings.
        """
        self.cols = [
            'z', 'zErr', 'rChi2', 'velDisp', 'velDispErr', 'extinction_r', 'petroMag_r', 
            'psfMag_r', 'psfMagErr_r', 'modelMag_u', 'modelMagErr_u', 
            'modelMag_g', 'modelMagErr_g', 'modelMag_r', 'modelMagErr_r', 
            'modelMag_i', 'modelMagErr_i', 'modelMag_z', 'modelMagErr_z', 
            'petroR50_r', 'petroR90_r', 'nii_6584_flux', 'nii_6584_flux_err', 
            'h_alpha_flux', 'h_alpha_flux_err', 'oiii_5007_flux', 
            'oiii_5007_flux_err', 'h_beta_flux', 'h_beta_flux_err', 
            'h_delta_flux', 'h_delta_flux_err', 'd4000', 'd4000_err', 
            'bptclass', 'lgm_tot_p50', 'sfr_tot_p50'
        ]
        cat_data = fetch_sdss_specgals()
        print(f"Loaded {cat_data.shape[0]} galaxies.")
        
        files = natsorted(glob.glob(f"{embeddings_fp}/{split}/*.parquet"))

        # 2. Define the file path mapping using the explicitly sorted lists
        data_files = {
            split: files
        }

        # 3. Load the files into a single DatasetDict
        ds = load_dataset("parquet", data_files=data_files)
        print(ds)

        df_sdss = pd.DataFrame(cat_data)
        print(df_sdss)
        df_sdss['merge_id'] = df_sdss['specObjID'].astype('int64')

        print(f"--- Processing '{split}' split ---")
        # Convert the current Hugging Face split to a Pandas DataFrame
        df_spender = ds[split].to_pandas()
        
        # Clean the Hugging Face IDs (strip the 'b', quotes, and spaces)
        df_spender['merge_id'] = df_spender['id'].astype(str).str.extract(r'(\d+)')[0].astype('int64')
        
        # Perform the master merge
        dset = pd.merge(df_spender, df_sdss, on='merge_id', how='inner')
        
        # Clean up by dropping the temporary merge column
        dset = dset.drop(columns=['merge_id']).rename(columns={'z_x': 'z'})
        
        print(f"Successfully merged {len(dset)} galaxies.")
        print("All splits successfully matched and aligned!")

        print(dset)

        dset = dset[dset[embed_type].notna()].reset_index(drop=True) # Drop NaN
        mask = ((dset[list(params)] != -9999.0).all(axis=1)) & (dset["mask_ratio"] != 1.) # Drop -99 Filler values
        dset = dset[mask].reset_index(drop=True)

        self.X = dset[embed_type].to_numpy()          # shape: (N, D)
        self.Y = dset[list(params)].to_numpy(dtype=np.float32)  # shape: (N, P)
        self.mask_ratio = dset["mask_ratio"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
        mask_ratio = self.mask_ratio[idx]
        return x, y, mask_ratio
    

class EmbeddingDataLoader(L.LightningDataModule):
    def __init__(
        self,
        datasets,
        batch_size=64,
        random_state=42,
        shuffle=True,
        num_workers=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.random_state = random_state
        if not num_workers:
            num_workers = os.cpu_count() - 1
            if num_workers > 16:  # limit worker allocation to stop
                # torch dataloader complaints. Need better way of doing this.
                self.num_workers = 16
            else:
                self.num_workers = num_workers
        else:
            self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, _stage=None):
        self.train_dataset = self.datasets["train"]
        self.val_dataset = self.datasets["val"]
        self.test_dataset = self.datasets["test"]

    def base_dataloader(self, dataset, split):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if split == "train" else False,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.base_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self.base_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self.base_dataloader(self.test_dataset, "test")


if __name__ == "__main__":

    '''
    dset_dict = {
        "train": EmbeddingDataset(
            embeddings_fp="/data/dtce-schmidt/phys2526/sdss_II/spender_I_flow_v2/embeddings/7655991_0",
            catalog_dset_name="spectra_catalog",
            split="train",
            embed_type="orig"
        ),
        "val": EmbeddingDataset(
            embeddings_fp="/data/dtce-schmidt/phys2526/sdss_II/spender_I_flow_v2/embeddings/7655991_0",
            catalog_dset_name="spectra_catalog",
            split="val",
            embed_type="orig"
        ),
        "test": EmbeddingDataset(
            embeddings_fp="/data/dtce-schmidt/phys2526/sdss_II/spender_I_flow_v2/embeddings/7655991_0",
            catalog_dset_name="spectra_catalog",
            split="test",
            embed_type="orig"
        )
    }
    '''
    dset_dict = {
        "train": JHU_MPA_EmbeddingDataset(
            embeddings_fp="/data/dtce-schmidt/phys2526/sdss_II/spender_I_flow_v2/embeddings/7655991_0",
            split="train",
            embed_type="orig"
        ),
        "val": JHU_MPA_EmbeddingDataset(
            embeddings_fp="/data/dtce-schmidt/phys2526/sdss_II/spender_I_flow_v2/embeddings/7655991_0",
            split="val",
            embed_type="orig"
        ),
        "test": JHU_MPA_EmbeddingDataset(
            embeddings_fp="/data/dtce-schmidt/phys2526/sdss_II/spender_I_flow_v2/embeddings/7655991_0",
            split="test",
            embed_type="orig"
        )
    }

    '''
    dataloader = EmbeddingDataLoader(
        datasets=dset_dict
    )
    dataloader.setup()
    df = dataloader.test_dataset
    '''
    #print(df.isna())
    for split in ["train", "val", "test"]:
        print(dset_dict[split])
        print(len(dset_dict[split]))

'''# Drop rows where ANY param == -99
mask = (dset[list(params)] != -99).all(axis=1)
dset = dset[mask].reset_index(drop=True)
'''
