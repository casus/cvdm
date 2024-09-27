from dataclasses import dataclass


@dataclass
class EvalConfig:
    """
    Configuration settings for the evaluation process during model training.

    Attributes:
        generation_timesteps (int): Number of timesteps to use for image generation.
        output_path (str): Directory where generated images and results will be saved.
        image_freq (float): Frequency (in terms of steps) at which images are generated for inspection.
        checkpoint_freq (int): Frequency (in steps) at which the model checkpoints are saved.
        log_freq (int): Frequency (in steps) at which loss logs are recorded.
        val_freq (int): Frequency (in steps) at which validation is performed.
        val_len (int): Number of samples used for validation.
    """

    generation_timesteps: int
    output_path: str
    image_freq: float
    checkpoint_freq: int
    log_freq: int
    val_freq: int
    val_len: int
