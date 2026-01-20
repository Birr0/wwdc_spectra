import copy

import lightning as L
import torch
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset


class WWDCDataset(Dataset):
    def __init__(self, dataset, return_catalog):
        self.dataset = dataset
        self.y_catalog = dataset.y_catalog
        self.return_catalog = return_catalog

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        item["y"] = self.process_catalog_entry(item["catalog"])
        if not self.return_catalog:
            item.pop("catalog")
        return item

    def process_catalog_entry(self, catalog):
        cat_entry = copy.deepcopy(catalog)
        for key in self.dataset.y_catalog["drop_variables"]:
            cat_entry.pop(key)

        for key in cat_entry:
            fn = self.y_catalog["variables"][key]["processing_fn"]
            if fn is not None:
                try:
                    cat_entry[key] = fn(cat_entry[key])
                except Exception as e:
                    msg = f"Error processing {key}: {e}"
                    raise ValueError(msg) from e

        if self.y_catalog["join_method"] == "concat":
            if len(cat_entry) > 0:
                return torch.cat(list(cat_entry.values()), dim=0)
            return torch.tensor([])
        elif self.y_catalog["join_method"] is None:
            return cat_entry
        msg = f"Invalid join_method: {self.y_catalog['join_method']}"
        raise ValueError(msg)


class WWDCDataLoader(L.LightningDataModule):
    def __init__(
        self,
        datasets,
        batch_size=8,
        val_split=0.0,
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
            self.num_workers = 1  # os.cpu_count() - 1
        else:
            self.num_workers = num_workers
        self.shuffle = shuffle

    def create_val_split(self, train_dataset):
        if self.val_split > 0.0 and self.val_split < 1.0:
            train_indices, val_indices = train_test_split(
                range(len(self.train_dataset)),
                test_size=self.val_split,
                random_state=self.random_state,  # Ensure reproducibility across runs
            )  # stratify=labels,
        else:
            msg = f"Invalid val_split: {self.val_split}. \
            val_split must be between 0. and 1."
            raise ValueError(msg)

        self.train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(train_dataset, val_indices)

    def setup(self, _stage=None):
        self.train_dataset = self.datasets["train"]
        self.test_dataset = self.datasets["test"]

        if self.datasets["val"] is not None:
            self.val_dataset = self.datasets["val"]

        else:
            if self.val_split > 0.0 and self.val_split < 1.0:
                self.create_val_split(
                    self.train_dataset
                )  # creates self.train_dataset, self.val_dataset

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
