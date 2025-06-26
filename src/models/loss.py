from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor


class BaseLoss:
    @abstractmethod
    def __call__(
        self,
        ufeats: Tensor,
        pos_ifeats: Tensor,
        neg_ifeats: Tensor,
        pad_mask: Tensor | None,
        **kwargs,
    ) -> Tensor:
        """
        ufeats: (B, ..., D)
        pos_ifeats: (B, ..., D)
        neg_ifeats: (B, ..., N, D)
        """
        pass

    @staticmethod
    def _apply_pad_mask(loss: Tensor, pad_mask: Tensor) -> Tensor:
        mask = (~pad_mask.unsqueeze(-1)).float()  # (B, ..., 1)
        loss = loss * mask

        return loss.sum() / mask.sum().clamp(min=1)


class BPRLoss(BaseLoss):
    def __call__(self, ufeats, pos_ifeats, neg_ifeats, pad_mask=None, **kwargs):
        pos_scores = (ufeats * pos_ifeats).sum(dim=-1, keepdim=True)
        neg_scores = torch.matmul(
            ufeats.unsqueeze(-2), neg_ifeats.transpose(-1, -2)
        ).squeeze(-2)

        loss = -F.logsigmoid(pos_scores - neg_scores)
        if pad_mask is not None:
            return self._apply_pad_mask(loss, pad_mask)
        else:
            return loss.mean()


class BCELoss(BaseLoss):
    def __call__(
        self, ufeats, pos_ifeats, neg_ifeats, pad_mask=None, **kwargs
    ) -> Tensor:
        pos_scores = (ufeats * pos_ifeats).sum(dim=-1, keepdim=True)
        neg_scores = torch.matmul(
            ufeats.unsqueeze(-2), neg_ifeats.transpose(-1, -2)
        ).squeeze(-2)
        logits = torch.cat([pos_scores, neg_scores], dim=-1)
        target = torch.zeros_like(logits, device=logits.device)
        target[..., 0] = 1

        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if pad_mask is not None:
            return self._apply_pad_mask(loss, pad_mask)
        else:
            return loss.mean()
