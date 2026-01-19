import os

import lightning as L
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset


class WWDCEmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, y_catalog):
        """
        Args:
            embedding_dir (str): The directory containing the embeddings.
        """
        self.embedding_dir = embedding_dir
        self.y_catalog = y_catalog

        if not self.data_exists:
            msg = f"Embedding data not found locally at {self.embedding_dir}."
            raise ValueError(msg)

        self.embedding_dset = HFDataset.load_from_disk(self.embedding_dir)

    @property
    def data_exists(self):
        return os.path.exists(self.embedding_dir)

    def __len__(self):
        return len(self.embedding_dset)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.embedding_dset[idx].items()}

class WWDCEmbeddingDataLoader(L.LightningDataModule):
    def __init__(
        self,
        datasets,
        batch_size=64,
        val_split=0.2,
        random_state=42,
        shuffle=True,
        num_workers=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.val_split = val_split
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
    test_dataset = WWDCEmbeddingDataset(
        "/data/dtce-schmidt/phys2526/\
        galaxy10_decals/galaxy10_VAE/embeddings/5287281/test",
        y_catalog=None,
    )
    print(test_dataset[0])