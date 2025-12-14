from abc import abstractmethod
from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from src.models.registry import get_loss_class

if TYPE_CHECKING:
    from src.config import Config
    from src.process.processor import TrainData


class BaseModel(nn.Module):
    @classmethod
    @abstractmethod
    def build(cls, config: "Config", data: "TrainData") -> "BaseModel":
        """Factory method to build model from config and data.

        Each model should implement this to extract its required arguments.
        """
        raise NotImplementedError

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
        loss_cls = get_loss_class(loss_fn)
        return loss_cls()
