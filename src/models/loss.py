from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.registry import register_loss


class BaseLossWithNegativeSamples:
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
        if loss.dim() == pad_mask.dim() + 1:
            mask = (~pad_mask.unsqueeze(-1)).float()  # (B, ..., 1)
        else:
            mask = (~pad_mask).float()
        loss = loss * mask

        return loss.sum() / mask.sum().clamp(min=1)


@register_loss("bpr")
class BPRLossWithNegativeSamples(BaseLossWithNegativeSamples):
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


@register_loss("bce")
class BCELossWithNegativeSamples(BaseLossWithNegativeSamples):
    def __call__(
        self, ufeats, pos_ifeats, neg_ifeats, pad_mask=None, **kwargs
    ) -> Tensor:
        pos_scores = (ufeats * pos_ifeats).sum(dim=-1, keepdim=True)
        neg_scores = torch.matmul(
            ufeats.unsqueeze(-2), neg_ifeats.transpose(-1, -2)
        ).squeeze(-2)
        logits = torch.cat([pos_scores, neg_scores], dim=-1)
        target = torch.zeros_like(logits, device=logits.device)
        target[..., 0] = 1.0

        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if pad_mask is not None:
            return self._apply_pad_mask(loss, pad_mask)
        else:
            return loss.mean()


@register_loss("ce")
class CELossWithNegativeSamples(BaseLossWithNegativeSamples):
    def __call__(
        self, ufeats, pos_ifeats, neg_ifeats, pad_mask=None, **kwargs
    ) -> Tensor:
        pos_scores = (ufeats * pos_ifeats).sum(dim=-1, keepdim=True)
        neg_scores = torch.matmul(
            ufeats.unsqueeze(-2), neg_ifeats.transpose(-1, -2)
        ).squeeze(-2)

        logits = torch.cat([pos_scores, neg_scores], dim=-1)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        target = torch.zeros(
            logits_flat.shape[0], dtype=torch.int64, device=logits_flat.device
        )

        loss = F.cross_entropy(logits_flat, target, reduction="none")
        loss = loss.reshape(logits.shape[:-1])
        if pad_mask is not None:
            return self._apply_pad_mask(loss, pad_mask)
        else:
            return loss.mean()
