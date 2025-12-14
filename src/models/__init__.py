# Import loss module to populate LOSS_REGISTRY
from src.models import loss as _loss  # noqa: F401

# Import model classes to populate MODEL_REGISTRY
from src.models.lightgcn import LightGCN  # noqa: F401
from src.models.mf import MatrixFactorization  # noqa: F401
from src.models.sasrec import SASRec  # noqa: F401
