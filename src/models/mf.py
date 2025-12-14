from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from src.models.base_model import BaseModel
from src.models.registry import register_model

if TYPE_CHECKING:
    from src.config import Config
    from src.process.processor import TrainData


@register_model("mf", model_type="cf")
class MatrixFactorization(BaseModel):
    @classmethod
    def build(cls, config: Config, data: TrainData) -> MatrixFactorization:
        return cls(
            n_users=data.n_users,
            n_items=data.n_items,
            emb_dim=config.emb_dim,
            loss_fn=config.loss_fn,
        )

    def __init__(self, n_users: int, n_items: int, emb_dim: int, loss_fn: str):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        self.loss_fn = self._set_loss_fn(loss_fn)

    def forward(self) -> tuple[Tensor, Tensor]:
        return self.user_emb.weight, self.item_emb.weight

    def calc_loss(self, uids: Tensor, pos_iids: Tensor, neg_iids: Tensor) -> Tensor:
        all_ufeats, all_ifeats = self.forward()
        ufeats = all_ufeats[uids]
        pos_ifeats = all_ifeats[pos_iids]
        neg_ifeats = all_ifeats[neg_iids]

        return self.loss_fn(ufeats, pos_ifeats, neg_ifeats)

    def recommend(self, uids: Tensor, k: int, **kwargs) -> Tensor:
        all_ufeats, all_ifeats = self.forward()
        ufeats = all_ufeats[uids]
        scores = ufeats @ all_ifeats.T
        _, indices = scores.topk(k, dim=-1)

        return indices
