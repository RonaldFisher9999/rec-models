import torch
import torch.nn as nn
from torch import Tensor

from src.models.base_model import BaseModel


class LightGraphConv(nn.Module):
    def forward(self, adj: Tensor, feats: Tensor) -> Tensor:
        return torch.sparse.mm(adj, feats)


class LightGCN(BaseModel):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int,
        n_layers: int,
        adj: Tensor,
        loss_fn: str,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.all_emb = nn.Embedding(n_users + n_items, emb_dim)
        self.n_layers = n_layers
        self.adj = adj
        self.conv_layers = nn.ModuleList([LightGraphConv() for _ in range(n_layers)])
        self.alpha = 1 / (n_layers + 1)
        self.loss_fn = self._set_loss_fn(loss_fn)
        self.ufeats = torch.empty(n_users, emb_dim)
        self.ifeats = torch.empty(n_items, emb_dim)

    def forward(self) -> tuple[Tensor, Tensor]:
        feats = self.all_emb.weight

        out = [feats]
        for conv in self.conv_layers:
            feats = conv(self.adj, feats)
            out.append(feats)
        all_feats = torch.stack(out, dim=0).sum(dim=0) * self.alpha

        return all_feats[: self.n_users, :], all_feats[self.n_users :, :]

    def calc_loss(self, uids: Tensor, pos_iids: Tensor, neg_iids: Tensor) -> Tensor:
        all_ufeats, all_ifeats = self.forward()
        ufeats = all_ufeats[uids]
        pos_ifeats = all_ifeats[pos_iids]
        neg_ifeats = all_ifeats[neg_iids]

        return self.loss_fn(ufeats, pos_ifeats, neg_ifeats)

    def recommend(self, uids: Tensor, k: int, need_update: bool, **kwargs) -> Tensor:
        if need_update:
            self.ufeats, self.ifeats = self.forward()
        ufeats = self.ufeats[uids]
        scores = ufeats @ self.ifeats.T
        _, indices = scores.topk(k, dim=-1)

        return indices
