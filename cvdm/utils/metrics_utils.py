from typing import Dict, Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def nmae(y_pred: np.ndarray, y_real: np.ndarray) -> float:
    nmae: float = np.sqrt(np.sum((y_pred - y_real) ** 2)) / np.sqrt(np.sum(y_real**2))
    return nmae


def calculate_metrics(
    y_pred_batch: np.ndarray,
    y_real_batch: np.ndarray,
) -> Dict[str, float]:
    y_pred_batch = np.array(y_pred_batch)
    y_real_batch = np.array(y_real_batch)

    if y_pred_batch.shape[3] > 1:
        channel_axis = 2
    else:
        channel_axis = None

    metrics = {
        "mse": np.mean((y_pred_batch - y_real_batch) ** 2),
        "mape": np.mean(np.abs((y_real_batch - y_pred_batch) / y_real_batch + 1e-10))
        * 100,
        "nmae": np.mean(
            [
                np.mean(np.abs(y_pred - y_real)) / np.mean(np.abs(y_real) + 1e-10)
                for y_pred, y_real in zip(y_pred_batch, y_real_batch)
            ]
        ),
        "psnr": np.mean(
            [
                peak_signal_noise_ratio(
                    np.squeeze(y_pred), np.squeeze(y_real), data_range=2
                )
                for y_pred, y_real in zip(y_pred_batch, y_real_batch)
            ]
        ),
        "ssim": np.mean(
            [
                structural_similarity(
                    np.squeeze(y_pred),
                    np.squeeze(y_real),
                    data_range=2,
                    channel_axis=channel_axis,
                )
                for y_pred, y_real in zip(y_pred_batch, y_real_batch)
            ]
        ),
    }

    return metrics
