from src.config import Config
from src.data.process import TrainData
from src.models.lightgcn import LightGCN
from src.models.mf import MatrixFactorization


def build_model(config: Config, data: TrainData):
    device = config.device
    if config.model == "mf":
        return MatrixFactorization(
            data.n_users, data.n_items, config.emb_dim, config.loss_fn
        ).to(device)
    elif config.model == "lightgcn":
        return LightGCN(
            data.n_users,
            data.n_items,
            config.emb_dim,
            config.n_layers,
            data.adj.to(device),
            config.loss_fn,
        ).to(device)
    else:
        raise NotImplementedError
