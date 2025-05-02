from src.config import Config
from src.data.process import TrainData
from src.models.mf import MatrixFactorization


def build_model(config: Config, data: TrainData):
    if config.model == "mf":
        return MatrixFactorization(
            data.n_users, data.n_items, config.emb_dim, config.loss_fn
        )
    else:
        raise NotImplementedError
