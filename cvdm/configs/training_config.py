from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    lr: float
    epochs: int
