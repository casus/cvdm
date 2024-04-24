from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DataConfig:
    dataset_path: str
    n_samples: int
    batch_size: int
    im_size: int



