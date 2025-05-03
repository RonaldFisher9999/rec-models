import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.config import Config
from src.data.process import TrainData
from src.models.base_model import BaseModel
from src.train.train_dataset import BaseDataset


class Trainer:
    def __init__(self, config: Config, data: TrainData, model: BaseModel):
        self.config = config
        self.model = model
        self.device = config.device
        self.loaders = self._set_loaders(data)
        self.optim = torch.optim.Adam(params=model.parameters(), lr=config.lr)
        self.patience = 0

    def train(self):
        logger.info("Start training.")
        for epoch in range(1, self.config.n_epochs + 1):
            logger.info(f"Epoch {epoch}")
            self._train_step()
            metrics = self._valid_step("val")
            if self._check_metrics(metrics):
                self._save()
                self.patience = 0
            else:
                self.patience += 1
                if self.patience == self.config.max_patience:
                    logger.info("Early stop training.")
                    break
        self._load()
        self._valid_step("test")

    def _set_loaders(self, data: TrainData) -> dict[str, DataLoader]:
        loaders = {}
        for mode in ["train", "val", "test"]:
            rating_data = getattr(data, mode)
            dataset = BaseDataset(rating_data, data.n_items, self.config.n_neg_samples)
            shuffle = mode == "train"
            pin_memory = self.device == "cuda"
            loaders[mode] = DataLoader(
                dataset, self.config.batch_size, shuffle=shuffle, pin_memory=pin_memory
            )

        return loaders

    def _train_step(self):
        pass

    def _valid_step(self, mode: str):
        pass

    def _check_metrics(self, metrics):
        pass

    def _save(self):
        pass

    def _load(self):
        pass
