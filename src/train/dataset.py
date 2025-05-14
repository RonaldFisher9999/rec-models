from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate


class TrainDataset(Dataset):
    def __init__(self, rating_data: pd.DataFrame, n_items: int, n_neg_samples: int):
        super().__init__()
        self.data = rating_data[["user", "item"]].values.astype("int64")
        self.n_items = n_items
        self.n_neg_samples = n_neg_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        uid, pos_iid = self.data[idx]
        neg_iids = self._sample_negs()

        return {"uids": uid, "pos_iids": pos_iid, "neg_iids": neg_iids}

    def _sample_negs(self):
        return torch.randint(0, self.n_items, (self.n_neg_samples,))

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return default_collate(batch)


class ValidDataset(Dataset):
    def __init__(self, rating_data: pd.DataFrame):
        super().__init__()
        self.data = self._convert_data(rating_data)

    def _convert_data(self, rating_data: pd.DataFrame) -> list[tuple[int, np.ndarray]]:
        data = (
            rating_data.groupby("user")["item"].apply(lambda x: x.to_numpy()).to_dict()
        )
        return list(data.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        uid, labels = self.data[idx]

        return {"labels": labels, "inputs": {"uids": uid}}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [data["labels"] for data in batch]
        inputs = [data["inputs"] for data in batch]

        return {"labels": labels, "inputs": default_collate(inputs)}
