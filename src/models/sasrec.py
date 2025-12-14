from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base_model import BaseModel
from src.models.registry import register_model

if TYPE_CHECKING:
    from src.config import Config
    from src.process.processor import TrainData


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout_p: float):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.head_dim = dim // n_heads
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, feats: Tensor, pad_mask: Tensor | None, causal: bool) -> Tensor:
        B, S, _ = feats.size()
        q, k, v = torch.chunk(self.qkv_proj(feats), 3, dim=-1)  # (B, S, D)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(
            1, 2
        )  # (B, N_h, S, D_h)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = self._build_mask(pad_mask, causal, S, B, q.dtype, q.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropout_p)
        out = out.transpose(1, 2).view(B, S, self.dim)  # (B, S, D)

        return self.out_proj(out)

    def _build_mask(
        self,
        pad_mask: Tensor | None,
        causal: bool,
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        fill_value = max(torch.finfo(dtype).min, -1e5)
        if pad_mask is not None:
            attn_mask = (
                pad_mask.unsqueeze(1)
                .unsqueeze(2)
                .expand(batch_size, 1, seq_len, seq_len)
            )
        else:
            attn_mask = torch.zeros(
                batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=device
            )
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), 1
            )
            causal_mask.unsqueeze(0).unsqueeze(1).expand(
                batch_size, 1, seq_len, seq_len
            )
            attn_mask = torch.logical_or(attn_mask, causal_mask)

        return attn_mask.to(dtype).masked_fill(attn_mask, fill_value)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_p: float):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.linear1 = nn.Linear(dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, dim)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = nn.ReLU()

    def forward(self, feats: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(feats))))


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, n_heads: int, dropout_p: float, causal: bool
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.mha = MultiHeadAttention(dim, n_heads, dropout_p)
        self.feed_forward = FeedForward(dim, hidden_dim, dropout_p)

    def forward(self, feats: Tensor, pad_mask: Tensor | None) -> Tensor:
        residual = feats
        feats = self.dropout1(self.mha(self.norm1(feats), pad_mask, self.causal))
        feats = feats + residual

        residual = feats
        feats = self.dropout2(self.feed_forward(self.norm2(feats)))
        feats = feats + residual

        return feats


@register_model("sasrec", model_type="sequential")
class SASRec(BaseModel):
    @classmethod
    def build(cls, config: Config, data: TrainData) -> SASRec:
        return cls(
            n_items=data.n_items,
            emb_dim=config.emb_dim,
            max_len=config.max_len,
            n_heads=config.n_heads,
            n_blocks=config.n_layers,
            dropout_p=config.dropout_p,
            padding_idx=data.n_items,
            causal=True,
            loss_fn=config.loss_fn,
        )

    def __init__(
        self,
        n_items: int,
        emb_dim: int,
        max_len: int,
        n_heads: int,
        n_blocks: int,
        dropout_p: float,
        padding_idx: int,
        causal: bool,
        loss_fn: str,
    ):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout_p = dropout_p
        self.padding_idx = padding_idx
        self.causal = causal
        self.emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.transformer_blokcs = nn.ModuleList(
            [
                TransformerBlock(emb_dim, emb_dim * 4, n_heads, dropout_p, causal)
                for _ in range(n_blocks)
            ]
        )
        self.loss_fn = self._set_loss_fn(loss_fn)

    def forward(self, iids: Tensor, pad_mask: Tensor) -> Tensor:
        feats = self._add_pos_emb(self.emb(iids))
        feats = self.dropout(feats)

        for block in self.transformer_blokcs:
            feats = block(feats, pad_mask)

        return feats

    def _add_pos_emb(self, feats: Tensor) -> Tensor:
        seq_len = feats.size(1)
        return feats + self.pos_emb.weight[:seq_len, :].unsqueeze(0)

    def calc_loss(
        self, pos_iids: Tensor, neg_iids: Tensor, pad_mask: Tensor, labels: Tensor
    ) -> Tensor:
        """
        pos_iids: (B, S)
        neg_iids: (B, N, S)
        pad_mask: (B, S)
        labels: (B, S)
        """
        ufeats = self.forward(pos_iids, pad_mask)  # (B, S, D)
        pos_ifeats = self.emb(labels)  # (B, S, D)
        neg_ifeats = self.emb(neg_iids[:, : self.max_len, :])  # (B, S, N, D)

        return self.loss_fn(ufeats, pos_ifeats, neg_ifeats, pad_mask)

    def recommend(self, pos_iids: Tensor, pad_mask: Tensor, k: int, **kwargs) -> Tensor:
        ufeats = self.forward(pos_iids, pad_mask)[:, -1, :]
        scores = ufeats @ self.emb.weight.T
        _, indices = scores.topk(k, dim=-1)

        return indices
