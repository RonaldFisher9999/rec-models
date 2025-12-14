from src.config import Config
from src.models.base_model import BaseModel
from src.models.registry import get_model_class
from src.process.processor import TrainData


def build_model(config: Config, data: TrainData) -> BaseModel:
    """Build model from config using registry.

    The model class is looked up from MODEL_REGISTRY and its `build`
    class method is called to construct the instance.
    """
    model_cls = get_model_class(config.model)
    return model_cls.build(config, data).to(config.device)
