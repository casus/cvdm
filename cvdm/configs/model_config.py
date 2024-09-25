from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration settings for the model architecture and initialization.

    Attributes:
        noise_model_type (str): The type of noise model used (unet or sr3).
        alpha (float): The scaling factor for L_gamma element of CVDM loss.
        snr_expansion_n (int): The expansion factor for the Taylor expansion of SNR.
        load_weights (str): Path to the pre-trained model weights to be loaded.
        load_mu_weights (str): Path to the pre-trained weights for the mean (mu) model, if applicable.
        zmd (bool): Indicates whether Zero-Mean Diffusion (ZMD) should be used in the model.
        inp_diff (bool): Indicates whether the model should predict just the noise to remove in each diffusion step or the image with the noise already removed.
    """

    noise_model_type: str
    alpha: float
    snr_expansion_n: int
    load_weights: str
    load_mu_weights: str
    zmd: bool
    diff_inp: bool
