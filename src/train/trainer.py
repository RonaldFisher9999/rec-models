import os
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data.process import TrainData
from src.models.base_model import BaseModel
from src.train.dataset import TrainDataset, ValidDataset


class Trainer:
    def __init__(self, config: Config, data: TrainData, model: BaseModel):
        self.config = config
        self.model = model
        self.device = config.device
        self.loaders = self._set_loaders(data)
        self.optim = torch.optim.Adam(params=model.parameters(), lr=config.lr)
        self.evaluator = Evaluator(config.eval_k)
        self.patience = 0
        self.target_metric = "ndcg_10"
        self.all_metrics = []
        self.best_score = 0.0
        self.checkpoint_path = self._set_checkpoint()

    def train(self):
        logger.info("Start training.")
        for epoch in range(1, self.config.n_epochs + 1):
            logger.info(f"Epoch {epoch}")
            train_loss = self._train_step()
            metrics = self._valid_step("val")
            self.all_metrics.append(metrics)
            if self._check_metrics(metrics):
                self._save(epoch, train_loss)
                self.patience = 0
            else:
                self.patience += 1
                if self.patience == self.config.max_patience:
                    logger.info("Early stop training.")
                    break
        logger.info("Load best model from checkpoint.")
        self._load()
        test_metrics = self._valid_step("test")
        metric, k = self.target_metric.split("_")
        logger.info(f"Test score: {test_metrics[metric][int(k)]:.3f}")

    def _set_loaders(self, data: TrainData) -> dict[str, DataLoader]:
        loaders = {}
        pin_memory = self.device == "cuda"
        for mode in ["train", "val", "test"]:
            rating_data = getattr(data, mode)
            if mode == "train":
                dataset = TrainDataset(
                    rating_data, data.n_items, self.config.n_neg_samples
                )
            else:
                if mode == "val":
                    dataset = ValidDataset(rating_data, data.train)
                else:
                    dataset = ValidDataset(rating_data, data.train, data.val)

            shuffle = mode == "train"
            loaders[mode] = DataLoader(
                dataset,
                self.config.batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                collate_fn=dataset.collate_fn,
            )

        return loaders

    def _set_checkpoint(self) -> str:
        ts = int(datetime.now().timestamp())
        train_name = f"{self.config.dataset}-{self.config.model}-{ts}"
        dir = os.path.join(self.config.checkpoint_dir, train_name)
        path = os.path.join(dir, self.config.checkpoint_file)
        logger.info(f"Save checkpoint at: {path}")
        os.makedirs(dir, exist_ok=True)

        return path

    def _train_step(self) -> float:
        self.model.train()

        total_loss = torch.zeros(1, device=self.device).squeeze_()
        for batch in tqdm(self.loaders["train"]):
            batch = self._to_device(batch)
            loss = self.model.calc_loss(**batch)
            total_loss += loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        total_loss = total_loss.item()
        logger.info(f"Train Loss: {total_loss:.3f}")

        return total_loss

    @torch.inference_mode()
    def _valid_step(self, mode: str) -> dict[str, dict[int, float]]:
        max_k = max(self.config.eval_k)
        logger.info(f"{mode.capitalize()} step.")
        self.model.eval()

        labels, seen, preds = [], [], []
        need_update = True  # for LightGCN
        for batch in tqdm(self.loaders[mode]):
            inputs = self._to_device(batch["model_input"])
            extra_k = max(x.size for x in batch["seen"])
            batch_preds = (
                self.model.recommend(
                    **inputs, k=max_k + extra_k, need_update=need_update
                )
                .detach()
                .cpu()
                .numpy()
            )
            need_update = False
            labels.extend(batch["labels"])
            seen.extend(batch["seen"])
            preds.extend([pred for pred in batch_preds])

        metrics = self.evaluator.calc_metrics(labels, preds, seen)
        logger.info(metrics)

        return metrics

    def _check_metrics(self, metrics: dict[str, dict[int, float]]) -> bool:
        metric, k = self.target_metric.split("_")
        current_score = metrics[metric][int(k)]
        if current_score > self.best_score:
            logger.info(
                f'Update best "{self.target_metric}" score {self.best_score:.3f} -> {current_score:.3f}'
            )
            self.best_score = current_score
            return True
        else:
            return False

    def _save(self, epoch: int, train_loss: float):
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optim_state": self.optim.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "metrics": self.all_metrics,
            "best_score": self.best_score,
            "patience": self.patience,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _load(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

    def _to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}


class Evaluator:
    def __init__(self, eval_k: list[int]):
        self.eval_k = sorted(eval_k)
        self.exclude_seen = False

    def calc_metrics(
        self,
        labels: list[np.ndarray],
        preds: list[np.ndarray],
        all_seen: list[np.ndarray],
    ) -> dict[str, dict[int, float]]:
        metrics = {
            "recall": {k: 0.0 for k in self.eval_k},
            "ndcg": {k: 0.0 for k in self.eval_k},
        }
        n_users = len(labels)
        for k in self.eval_k:
            recall_total, ndcg_total = 0, 0
            discounts = 1.0 / np.log2(np.arange(k, dtype=np.float32) + 2.0)
            for label, pred, seen in zip(labels, preds, all_seen):
                if self.exclude_seen:
                    pred = pred[~np.isin(pred, seen)][:k]
                else:
                    pred = pred[:k]
                hits = np.isin(pred, label).astype("float")

                recall_total += hits.sum() / len(label)

                dcg = (hits * discounts).sum()
                idcg = discounts[: min(len(label), k)].sum()
                ndcg_total += dcg / idcg
            metrics["recall"][k] = recall_total / n_users
            metrics["ndcg"][k] = ndcg_total / n_users

        return metrics
