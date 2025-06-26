import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import Tensor

from src.config import Config


@dataclass
class TrainData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    adj: Tensor | None


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def _load_raw_data(self) -> pd.DataFrame:
        if self.config.dataset == "movielens":
            return load_movielens("dataset/movielens/")
        else:
            raise NotImplementedError()

    def _split_data(
        self, ratings: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.config.split_method in ("ratio", "leave_one_out"):
            total_train, test = split_by_col(
                ratings, "user", self.config.split_method, self.config.test_ratio
            )
            train, val = split_by_col(
                total_train, "user", self.config.split_method, self.config.val_ratio
            )
            return (
                train.reset_index(drop=True),
                val.reset_index(drop=True),
                test.reset_index(drop=True),
            )
        else:
            raise NotImplementedError()

    def process(self) -> TrainData:
        ratings = self._load_raw_data()
        ratings = filter_by_cnt(ratings, "item", self.config.min_item_cnt)
        ratings = filter_by_cnt(ratings, "user", self.config.min_item_cnt)
        ratings = map_to_idx(ratings, ["user", "item"])
        ratings = ratings.sort_values("timestamp", ignore_index=True)
        n_users, n_items = ratings["user"].nunique(), ratings["item"].nunique()
        train, val, test = self._split_data(ratings)
        if self.config.model in ["lightgcn"]:
            adj = build_adj_mat(train, n_users, n_items)
        else:
            adj = None

        return TrainData(train, val, test, n_users, n_items, adj)


def load_movielens(data_dir: str) -> pd.DataFrame:
    logger.info("Loading Movielens-1m dataset...")
    path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        path, sep="::", engine="python", header=None, encoding="latin1"
    ).iloc[:, [0, 1, 3]]
    df.columns = ["user", "item", "timestamp"]
    # TODO need to add random noise to gurantee same order
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    return df


def filter_by_cnt(df: pd.DataFrame, col: str, cnt: int) -> pd.DataFrame:
    id_counts = df[col].value_counts()
    valid_ids = set(id_counts[id_counts >= cnt].index.values)
    logger.info(f'Remove "{col}" less than {cnt} counts.')
    logger.info(f"Size reduced: {len(id_counts)} -> {len(valid_ids)}")

    return df[df[col].isin(valid_ids)]


def map_to_idx(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        unique_ids = df[col].unique()
        mapper = {v: i for i, v in enumerate(unique_ids)}
        df[col] = df[col].map(mapper)

    return df


def split_by_col(
    df: pd.DataFrame, col: str, method: str, ratio: float | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby(col)
    if method == "ratio":
        indices = []
        for _, g in grouped:
            size = max(1, int(len(g) * ratio))
            indices.extend(g.tail(size).index.tolist())
        return df.loc[~df.index.isin(set(indices)), :], df.loc[indices, :]
    elif method == "leave_one_out":
        tails = grouped.tail(1)
        indices = tails.index
        return df.loc[~df.index.isin(set(indices)), :], tails
    else:
        raise NotImplementedError()


def build_adj_mat(df: pd.DataFrame, n_users: int, n_items: int) -> Tensor:
    user_idx = torch.from_numpy(df["user"].to_numpy(np.int64))
    item_idx = torch.from_numpy(df["item"].to_numpy(np.int64) + n_users)
    u2i_idx = torch.stack([user_idx, item_idx], dim=0)
    i2u_idx = torch.stack([item_idx, user_idx], dim=0)

    edge_idx = torch.concat([u2i_idx, i2u_idx], dim=1)
    values = torch.ones(edge_idx.shape[1], dtype=torch.float32)
    n = n_users + n_items
    adj = torch.sparse_coo_tensor(edge_idx, values, size=(n, n))

    degree = torch.sparse.sum(adj, dim=0).to_dense()
    norm_degree = degree.pow(-0.5)
    # norm_degree[norm_degree == float('inf')] = 0.0
    rows, cols = edge_idx[0], edge_idx[1]
    norm_values = norm_degree[rows] * norm_degree[cols]

    return torch.sparse_coo_tensor(edge_idx, norm_values, size=(n, n)).to_sparse_csc()
