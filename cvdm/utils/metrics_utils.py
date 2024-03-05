from typing import Dict, Optional
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def nmae(y_pred: np.ndarray, y_real: np.ndarray) -> float:
    nmae: float = np.sqrt(np.sum((y_pred - y_real) ** 2)) / np.sqrt(np.sum(y_real**2))
    return nmae


def dice(y_pred_masks: np.ndarray, y_real_masks: np.ndarray) -> float:
    intersection = np.sum(y_pred_masks * y_real_masks)
    dice: float = intersection * 2.0 / (np.sum(y_real_masks) + np.sum(y_pred_masks))
    return dice


def iou(y_pred_masks: np.ndarray, y_real_masks: np.ndarray) -> float:
    intersection = np.sum(y_pred_masks * y_real_masks)
    union = np.sum(y_real_masks + y_pred_masks) - intersection
    iou: float = intersection / union
    return iou


def calculate_metrics(
    y_pred_batch: np.ndarray,
    y_real_batch: np.ndarray,
    y_pred_masks_batch: Optional[np.ndarray] = None,
    y_real_masks_batch: Optional[np.ndarray] = None,
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

    # Optional mask-related metrics
    if y_pred_masks_batch is not None and y_real_masks_batch is not None:
        metrics["dice"] = np.mean(dice(y_pred_masks_batch, y_real_masks_batch))
        metrics["iou"] = np.mean(iou(y_pred_masks_batch, y_real_masks_batch))

    return metrics
