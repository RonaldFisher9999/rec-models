from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.process.processor import TrainData
from src.train.dataset import (
    CFTrainDataset,
    CFValidDataset,
    SequentialTrainDataset,
    SequentialValidDataset,
)


def build_dataloaders(config: Config, data: TrainData) -> dict[str, DataLoader]:
    datasets: dict[str, Dataset] = {}
    loaders: dict[str, DataLoader] = {}
    pin_memory = config.device == "cuda"
    model_type = config.model_type
    if model_type == "cf":
        datasets["train"] = CFTrainDataset(config, data)
        for mode in ["val", "test"]:
            datasets[mode] = CFValidDataset(config, data, mode)
    elif model_type == "sequential":
        padding_idx = data.n_items
        datasets["train"] = SequentialTrainDataset(config, data, padding_idx)
        for mode in ["val", "test"]:
            datasets[mode] = SequentialValidDataset(config, data, padding_idx, mode)
    else:
        raise NotImplementedError()

    for mode, dataset in datasets.items():
        shuffle = mode == "train"
        loaders[mode] = DataLoader(
            dataset,
            config.batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=dataset.collate_fn,
        )

    return loaders
