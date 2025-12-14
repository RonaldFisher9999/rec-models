from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from src.models.base_model import BaseModel
    from src.models.loss import BaseLossWithNegativeSamples

T = TypeVar("T")

MODEL_REGISTRY: dict[str, dict[str, Any]] = {}
LOSS_REGISTRY: dict[str, type["BaseLossWithNegativeSamples"]] = {}


def register_model(name: str, model_type: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a model class.

    Args:
        name: Model name used in CLI (e.g., "mf", "lightgcn", "sasrec")
        model_type: Model type for dataset selection ("cf" or "sequential")

    Example:
        @register_model("mf", model_type="cf")
        class MatrixFactorization(BaseModel):
            ...
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = {"cls": cls, "type": model_type}
        return cls

    return decorator


def register_loss(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a loss function class.

    Args:
        name: Loss name used in CLI (e.g., "bpr", "bce", "ce")

    Example:
        @register_loss("bpr")
        class BPRLossWithNegativeSamples(BaseLossWithNegativeSamples):
            ...
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered")
        LOSS_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> type["BaseModel"]:
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]["cls"]


def get_model_type(name: str) -> str:
    """Get model type (cf/sequential) by model name."""
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]["type"]


def get_loss_class(name: str) -> type["BaseLossWithNegativeSamples"]:
    """Get loss class by name."""
    if name not in LOSS_REGISTRY:
        available = list(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss '{name}'. Available: {available}")
    return LOSS_REGISTRY[name]
