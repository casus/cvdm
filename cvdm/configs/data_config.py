from dataclasses import dataclass


@dataclass
class DataConfig:
    """
    Configuration for the dataset.

    Attributes:
        dataset_path (str): Path to the dataset directory. For NpyDataloader it is expected that it will contain x.npy and y.npy files.
        For PhasePolychromeDataloader it should contain y.npy. For the rest of dataloaders, the directory should include .JPEG images.
        n_samples (int): Number of samples to use from the dataset.
        batch_size (int): Number of samples per batch during training.
        im_size (int): The size of the patches of images (both height and width) to use.
    """

    dataset_path: str
    n_samples: int
    batch_size: int
    im_size: int
