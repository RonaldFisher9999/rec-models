import os
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from src.config import Config


@dataclass
class TrainData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def _load_raw_data(self) -> pd.DataFrame:
        if self.config.dataset == "movielens":
            return load_movielens("dataset/movielens/")
        else:
            raise NotImplementedError()

    def _process_data(self, ratings: pd.DataFrame) -> pd.DataFrame:
        ratings = filter_by_cnt(ratings, "item", self.config.min_item_cnt)
        ratings = filter_by_cnt(ratings, "user", self.config.min_item_cnt)
        ratings = map_to_idx(ratings, ["user", "item"])

        return ratings.sort_values("timestamp", ignore_index=True).drop(
            columns=["timestamp"]
        )

    def _split_data(
        self, ratings: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.config.split_method == "random":
            total_train, test = split_by_col(ratings, "user", self.config.test_ratio)
            train, val = split_by_col(total_train, "user", self.config.val_ratio)
            return (
                train.reset_index(drop=True),
                val.reset_index(drop=True),
                test.reset_index(drop=True),
            )
        else:
            raise NotImplementedError()

    def process(self) -> TrainData:
        ratings = self._load_raw_data()
        ratings = self._process_data(ratings)
        n_users, n_items = ratings["user"].nunique(), ratings["item"].nunique()
        train, val, test = self._split_data(ratings)

        return TrainData(train, val, test, n_users, n_items)


def load_movielens(data_dir: str) -> pd.DataFrame:
    logger.info("Loading Movielens-1m dataset...")
    path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        path, sep="::", engine="python", header=None, encoding="latin1"
    ).iloc[:, [0, 1, 3]]
    df.columns = ["user", "item", "timestamp"]
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
    df: pd.DataFrame, col: str, ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby(col)
    indices = []
    for _, g in grouped:
        size = max(1, int(len(g) * ratio))
        indices.extend(g.tail(size).index.tolist())
    return df.loc[~df.index.isin(set(indices)), :], df.loc[indices, :]
