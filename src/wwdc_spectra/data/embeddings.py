import os

import numpy as np 
import lightning as L
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from wwdc_spectra.data.cross_match import merge_matched_cat_with_embeddings

class EmbeddingDataset(Dataset):
    def __init__(
        self, 
        embedding_dset_name,
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
            matched_cat=catalog_dset_name,
            embeddings=embedding_dset_name,
            key="id",
            join="left outer",
            split=split
        )
        
        dset = dset[dset[embed_type].notna()].reset_index(drop=True) # Drop NaN
        mask = (dset[list(params)] != -99).all(axis=1) # Drop -99 Filler values
        dset = dset[mask].reset_index(drop=True)

        self.X = np.stack(dset[embed_type].to_numpy())          # shape: (N, D)
        self.Y = dset[list(params)].to_numpy(dtype=np.float32)  # shape: (N, P)

    '''
    Add local option later
    @property
    def data_exists(self):
        return os.path.exists(self.embedding_dir)
    '''

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.Y[idx]).float()
        return x, y

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
    dset = EmbeddingDataset(
        embedding_dset_name="spender-I-vf-0",
        catalog_dset_name="spectra_catalog",
        split="train"
    )
    print(dset)
    print(len(dset))
    '''
    dset_dict = {
        "train": EmbeddingDataset(
            embedding_dset_name="spender-I-vf-0",
            catalog_dset_name="spectra_catalog",
            split="train",
            embed_type="orig"
        ),
        "val": EmbeddingDataset(
            embedding_dset_name="spender-I-vf-0",
            catalog_dset_name="spectra_catalog",
            split="val",
            embed_type="orig"
        ),
        "test": EmbeddingDataset(
            embedding_dset_name="spender-I-vf-0",
            catalog_dset_name="spectra_catalog",
            split="test",
            embed_type="orig"
        )
    }
    dataloader = EmbeddingDataLoader(
        datasets=dset_dict
    )
    print(dataloader)
    dataloader.setup()
    df = dataloader.test_dataset.X
    #print(df.isna())


'''# Drop rows where ANY param == -99
mask = (dset[list(params)] != -99).all(axis=1)
dset = dset[mask].reset_index(drop=True)
'''
