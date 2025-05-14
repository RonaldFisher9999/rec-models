from abc import abstractmethod

import torch.nn as nn
from torch import Tensor

from src.models.loss import BCELoss, BPRLoss


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def calc_loss(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def recommend(self) -> Tensor:
        raise NotImplementedError

    def _set_loss_fn(self, loss_fn: str):
        if loss_fn == "bpr":
            return BPRLoss()
        elif loss_fn == "bce":
            return BCELoss()
        else:
            raise NotImplementedError
