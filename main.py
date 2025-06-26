from loguru import logger

from src.config import config_parser
from src.models.utils import build_model
from src.process.processor import DataProcessor
from src.train.trainer import Trainer
from src.train.utils import build_dataloaders


def main():
    config = config_parser()
    logger.info(config)

    processor = DataProcessor(config)
    data = processor.process()

    model = build_model(config, data)
    dataloaders = build_dataloaders(config, data)

    trainer = Trainer(config, model, dataloaders)
    trainer.train()


if __name__ == "__main__":
    main()
