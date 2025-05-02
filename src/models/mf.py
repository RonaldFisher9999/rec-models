import torch.nn as nn
from torch import Tensor

from src.models.base_model import BaseModel


class MatrixFactorization(BaseModel):
    def __init__(self, n_users: int, n_items: int, emb_dim: int, loss_fn: str):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        self.loss_fn = self._set_loss_fn(loss_fn)

    def forward(self):
        return self.user_emb.weight, self.item_emb.weight

    def calc_loss(self, uids: Tensor, pos_iids: Tensor, neg_iids: Tensor) -> Tensor:
        all_ufeats, all_ifeats = self.forward()
        ufeats = all_ufeats[uids]
        pos_ifeats = all_ifeats[pos_iids]
        neg_ifeats = all_ifeats[neg_iids]

        return self.loss_fn(ufeats, pos_ifeats, neg_ifeats)

    def recommend(self):
        return
