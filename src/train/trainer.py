from src.config import Config
from src.data.process import TrainData


class Trainer:
    def __init__(self, config: Config, data: TrainData, model):
        self.config = config
        self.model = model

    def prepare(self, data: TrainData):
        pass

    def train(self):
        pass
