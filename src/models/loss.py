from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor


class BaseLoss:
    @abstractmethod
    def __call__(
        self, ufeats: Tensor, pos_ifeats: Tensor, neg_ifeats: Tensor
    ) -> Tensor:
        pass


class BPRLoss(BaseLoss):
    def __call__(self, ufeats, pos_ifeats, neg_ifeats):
        pos_scores = (ufeats * pos_ifeats).sum(dim=-1, keepdim=True)
        neg_scores = torch.matmul(
            ufeats.unsqueeze(1), neg_ifeats.transpose(1, 2)
        ).squeeze(1)

        return F.logsigmoid(pos_scores - neg_scores).mean()


class BCELoss(BaseLoss):
    def __call__(self, ufeats, pos_ifeats, neg_ifeats):
        pos_scores = (ufeats * pos_ifeats).sum(dim=-1, keepdim=True)
        neg_scores = torch.matmul(
            ufeats.unsqueeze(1), neg_ifeats.transpose(1, 2)
        ).squeeze(1)
        logits = torch.cat([pos_scores, neg_scores], dim=1)
        target = torch.zeros_like(logits, device=logits.device)
        target[..., 0] = 1

        return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
