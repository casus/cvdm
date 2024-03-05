from dataclasses import dataclass


@dataclass
class EvalConfig:
    generation_timesteps: int
    output_path: str
    image_freq: float
    checkpoint_freq: int
    log_freq: int
    val_freq: int
    val_len: int
