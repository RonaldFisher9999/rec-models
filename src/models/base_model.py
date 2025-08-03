from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from src.models.loss import (
    BCELossWithNegativeSamples,
    BPRLossWithNegativeSamples,
    CELossWithNegativeSamples,
)


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calc_loss(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def recommend(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def _set_loss_fn(self, loss_fn: str):
        if loss_fn == "bpr":
            return BPRLossWithNegativeSamples()
        elif loss_fn == "bce":
            return BCELossWithNegativeSamples()
        elif loss_fn == "ce":
            return CELossWithNegativeSamples()
        else:
            raise NotImplementedError
