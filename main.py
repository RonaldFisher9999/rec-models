from loguru import logger

from src.config import config_parser


def main():
    config = config_parser()
    logger.info(config)


if __name__ == "__main__":
    main()
