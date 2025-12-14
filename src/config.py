import argparse
from dataclasses import asdict, dataclass

# Import models package to populate the registry before parsing args
import src.models  # noqa: F401
from src.models.registry import (
    LOSS_REGISTRY,
    MODEL_REGISTRY,
    get_model_type,
)


@dataclass
class Config:
    dataset: str
    model: str
    seed: int
    min_user_cnt: int
    min_item_cnt: int
    split_method: str
    test_ratio: float
    val_ratio: float
    max_len: int
    n_neg_samples: int
    loss_fn: str
    n_layers: int
    emb_dim: int
    n_heads: int
    dropout_p: float
    batch_size: int
    lr: float
    n_epochs: int
    device: str
    checkpoint_dir: str
    checkpoint_file: str
    eval_k: list[int]
    max_patience: int
    model_type: str | None = None

    def __post_init__(self):
        self.model_type = get_model_type(self.model)

    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in asdict(self).items()])


def config_parser() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["movielens"], default="movielens"
    )
    parser.add_argument(
        "--model", type=str, choices=list(MODEL_REGISTRY.keys()), default="mf"
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--min_user_cnt", type=int, default=5)
    parser.add_argument("--min_item_cnt", type=int, default=5)
    parser.add_argument(
        "--split_method",
        type=str,
        choices=["ratio", "leave_one_out"],
        default="ratio",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--n_neg_samples", type=int, default=10)
    parser.add_argument(
        "--loss_fn", type=str, choices=list(LOSS_REGISTRY.keys()), default="ce"
    )
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout_p", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint.pt")
    parser.add_argument("--eval_k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--max_patience", type=int, default=5)

    args = parser.parse_args()
    config = Config(**vars(args))

    return config
