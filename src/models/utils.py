from src.config import Config
from src.data.process import TrainData
from src.models import LightGCN, MatrixFactorization, SASRec


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
    elif config.model == "sasrec":
        return SASRec(
            data.n_items,
            config.emb_dim,
            config.max_len,
            config.n_heads,
            config.n_layers,
            config.dropout_p,
            data.n_items,
            True,
            config.loss_fn,
        ).to(device)
    else:
        raise NotImplementedError()
