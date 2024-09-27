from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration settings for the model training process.

    Attributes:
        lr (float): Learning rate for the optimizer during training.
        epochs (int): Number of complete passes through the training dataset.
    """

    lr: float
    epochs: int
