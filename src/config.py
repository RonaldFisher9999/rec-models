import argparse
from dataclasses import dataclass, field


@dataclass
class Config:
    dataset: str = "movielens"
    model: str = "lightgcn"
    seed: int = 100
    min_user_cnt: int = 5
    min_item_cnt: int = 5
    split_method: str = "random"
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    n_neg_samples: int = 5
    max_seq_len: int = 50
    loss_fn: str = "bpr"
    n_layers: int = 3
    emb_dim: int = 64
    batch_size: int = 1024
    lr: float = 0.005
    n_epochs: int = 20
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints/"
    eval_k: list[int] = field(default_factory=lambda: [10, 20])


def config_parser() -> Config:
    default = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["movielens"], default=default.dataset
    )
    parser.add_argument("--model", type=str, choices=["mf"], default=default.model)
    parser.add_argument("--seed", type=int, default=default.seed)
    parser.add_argument("--min_user_cnt", type=int, default=default.min_user_cnt)
    parser.add_argument("--min_item_cnt", type=int, default=default.min_item_cnt)
    parser.add_argument(
        "--split_method", type=str, choices=["random"], default=default.split_method
    )
    parser.add_argument("--test_ratio", type=float, default=default.test_ratio)
    parser.add_argument("--val_ratio", type=float, default=default.val_ratio)
    parser.add_argument("--n_neg_samples", type=int, default=default.n_neg_samples)
    parser.add_argument("--max_seq_len", type=int, default=default.max_seq_len)
    parser.add_argument(
        "--loss_fn", type=str, choices=["bpr", "bce"], default=default.loss_fn
    )
    parser.add_argument("--n_layers", type=int, default=default.n_layers)
    parser.add_argument("--emb_dim", type=int, default=default.emb_dim)
    parser.add_argument("--batch_size", type=int, default=default.batch_size)
    parser.add_argument("--lr", type=float, default=default.lr)
    parser.add_argument("--n_epochs", type=int, default=default.n_epochs)
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default=default.device
    )
    parser.add_argument("--checkpoint_dir", type=str, default=default.checkpoint_dir)
    parser.add_argument("--eval_k", type=int, nargs="+", default=default.eval_k)

    args = parser.parse_args()
    config = Config(**vars(args))

    return config
