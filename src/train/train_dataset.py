import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
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
