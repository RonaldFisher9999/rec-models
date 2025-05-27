from typing import Any

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
    def __init__(self, ratings: pd.DataFrame, *seen: pd.DataFrame):
        super().__init__()
        self.data = self._convert_data(ratings, *seen)

    def _convert_data(
        self, ratings: pd.DataFrame, *seen: pd.DataFrame
    ) -> list[dict[str, Any]]:
        seen_grouped = (
            pd.concat(seen, axis=0)[["user", "item"]]
            .groupby("user")["item"]
            .apply(lambda x: x.to_numpy())
            .rename("seen")
        )
        ratings_grouped = (
            ratings[["user", "item"]]
            .groupby("user")["item"]
            .apply(lambda x: x.to_numpy())
            .rename("labels")
        )
        return (
            pd.merge(ratings_grouped, seen_grouped, on="user")
            .reset_index()
            .to_dict(orient="records")
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        return {
            "labels": row["labels"],
            "seen": row["seen"],
            "model_input": {"uids": row["user"]},
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [data["labels"] for data in batch]
        seen = [data["seen"] for data in batch]
        model_input = [data["model_input"] for data in batch]

        return {
            "labels": labels,
            "seen": seen,
            "model_input": default_collate(model_input),
        }
