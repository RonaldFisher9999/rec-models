from loguru import logger

from src.config import config_parser
from src.data.process import DataProcessor
from src.models.utils import build_model
from src.train.trainer import Trainer


def main():
    config = config_parser()
    logger.info(config)

    processor = DataProcessor(config)
    data = processor.process()

    model = build_model(config, data)

    trainer = Trainer(config, data, model)
    trainer.train()


if __name__ == "__main__":
    main()
