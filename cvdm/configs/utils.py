from typing import Any, Dict, Optional

import yaml

from cvdm.configs.data_config import DataConfig
from cvdm.configs.eval_config import EvalConfig
from cvdm.configs.model_config import ModelConfig
from cvdm.configs.neptune_config import NeptuneConfig
from cvdm.configs.training_config import TrainingConfig


def load_config_from_yaml(yaml_file: str) -> Dict[str, Any]:
    with open(yaml_file, "r") as file:
        config_data: Dict[str, Any] = yaml.safe_load(file)
    return config_data


def create_model_config(config_data: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(**config_data["model"])


def create_training_config(config_data: Dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(**config_data["training"])


def create_data_config(config_data: Dict[str, Any]) -> DataConfig:
    return DataConfig(**config_data["data"])


def create_eval_config(config_data: Dict[str, Any]) -> EvalConfig:
    return EvalConfig(**config_data["eval"])


def create_neptune_config(config_data: Dict[str, Any]) -> Optional[NeptuneConfig]:
    if "neptune" in config_data:
        return NeptuneConfig(**config_data["neptune"])
    else:
        return None
