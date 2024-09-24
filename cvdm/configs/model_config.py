from dataclasses import dataclass


@dataclass
class ModelConfig:
    noise_model_type: str
    alpha: float
    snr_expansion_n: int
    load_weights: str
    zmd: bool
