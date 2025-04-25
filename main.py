from loguru import logger

from src.config import config_parser
from src.data.process import DataProcessor
from src.train.trainer import Trainer


def main():
    config = config_parser()
    logger.info(config)

    processor = DataProcessor(config)
    trainer = Trainer(config)

    data = processor.process()
    trainer.prepare(data)
    trainer.train()


if __name__ == "__main__":
    main()
