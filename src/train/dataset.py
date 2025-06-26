from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate

from src.config import Config
from src.process.processor import TrainData


class CFTrainDataset(Dataset):
    def __init__(self, config: Config, data: TrainData):
        super().__init__()
        self.n_items = data.n_items
        self.n_neg_samples = config.n_neg_samples
        self.data = self._convert_data(data.train)

    def _convert_data(self, train_df: pd.DataFrame) -> list[dict[str, Any]]:
        return train_df[["user", "item"]].to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        record = self.data[idx]
        uid, pos_iid = record["user"], record["item"]
        neg_iids = self._sample_negs()

        return {"uids": uid, "pos_iids": pos_iid, "neg_iids": neg_iids}

    def _sample_negs(self):
        return torch.randint(0, self.n_items, (self.n_neg_samples,))

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return default_collate(batch)


class CFValidDataset(Dataset):
    def __init__(self, config: Config, data: TrainData, mode: str):
        super().__init__()
        if mode == "val":
            label = data.val
            seen = data.train
        else:
            label = data.test
            seen = pd.concat([data.train, data.val], axis=0)
        self.data = self._convert_data(label, seen)

    def _convert_data(
        self, label_df: pd.DataFrame, seen_df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        seen_grouped = (
            seen_df[["user", "item"]]
            .groupby("user")["item"]
            .apply(lambda x: x.to_numpy())
            .rename("seen")
        )
        label_grouped = (
            label_df[["user", "item"]]
            .groupby("user")["item"]
            .apply(lambda x: x.to_numpy())
            .rename("labels")
        )
        return (
            pd.merge(label_grouped, seen_grouped, on="user")
            .reset_index()
            .to_dict(orient="records")
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        record = self.data[idx]
        return {
            "labels": record["labels"],
            "seen": record["seen"],
            "model_input": {"uids": record["user"]},
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


class SequentialTrainDataset(Dataset):
    def __init__(
        self,
        config: Config,
        data: TrainData,
        padding_idx: int,
    ):
        super().__init__()
        self.max_len = config.max_len + 1
        self.padding_idx = padding_idx
        self.paddings = np.full(config.max_len, self.padding_idx)
        self.n_items = data.n_items
        self.n_neg_samples = config.n_neg_samples
        self.data = self._convert_data(data.train)

    def _convert_data(self, train_df: pd.DataFrame) -> list[dict[str, Any]]:
        return (
            train_df[["user", "item", "timestamp"]]
            # .sort_values('timestamp')
            .groupby("user")["item"]
            .apply(lambda x: x.to_numpy()[-self.max_len :])
            .rename("seq")
            .reset_index()
            .to_dict(orient="records")
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        record = self.data[idx]
        seq = record["seq"]
        seq = self._apply_paddings(seq)
        pos_iids = seq[:-1]
        labels = seq[1:]
        pad_mask = pos_iids == self.padding_idx
        neg_iids = self._sample_negs()

        return {
            "pos_iids": pos_iids,
            "neg_iids": neg_iids,
            "pad_mask": pad_mask,
            "labels": labels,
        }

    def _apply_paddings(self, seq: np.ndarray) -> np.ndarray:
        return np.append(self.paddings, seq)[-self.max_len :]

    def _sample_negs(self) -> np.ndarray:
        return np.random.randint(0, self.n_items, (self.max_len, self.n_neg_samples))

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return default_collate(batch)


class SequentialValidDataset(Dataset):
    def __init__(self, config: Config, data: TrainData, padding_idx: int, mode: str):
        super().__init__()
        self.max_len = config.max_len
        self.padding_idx = padding_idx
        self.paddings = np.full(config.max_len, self.padding_idx)
        self.n_items = data.n_items
        self.n_neg_samples = config.n_neg_samples
        if mode == "val":
            label = data.val
            seen = data.train
        else:
            label = data.test
            seen = pd.concat([data.train, data.val], axis=0)
        self.data = self._convert_data(label, seen)

    def _convert_data(
        self, label_df: pd.DataFrame, seen_df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        seen_grouped = (
            seen_df[["user", "item", "timestamp"]]
            # .sort_values('timestamp')
            .groupby("user")["item"]
            .apply(lambda x: x.to_numpy())
            .rename("seen")
        )
        label_grouped = (
            label_df[["user", "item"]]
            .groupby("user")["item"]
            .apply(lambda x: x.to_numpy())
            .rename("labels")
        )
        return (
            pd.merge(label_grouped, seen_grouped, on="user")
            .reset_index()
            .to_dict(orient="records")
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        record = self.data[idx]
        seen = record["seen"]
        pos_iids = self._apply_paddings(seen)
        pad_mask = pos_iids == self.padding_idx

        return {
            "labels": record["labels"],
            "seen": seen,
            "model_input": {"pos_iids": pos_iids, "pad_mask": pad_mask},
        }

    def _apply_paddings(self, seq: np.ndarray) -> np.ndarray:
        return np.append(self.paddings, seq)[-self.max_len :]

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
