from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from neptune import Run
from neptune.types import File
from skimage.util import montage
from tensorflow.keras.models import Model
from tqdm import tqdm

from cvdm.utils.metrics_utils import calculate_metrics


def ddpm_obtain_sr_img(
    x: np.ndarray,
    timesteps_test: int,
    noise_model: Model,
    schedule_model: Model,
    out_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if out_shape == None:
        out_shape = x.shape
    assert out_shape is not None
    pred_sr = np.random.normal(0, 1, out_shape)

    alpha_vec = np.zeros(out_shape + (timesteps_test,))
    for t in tqdm(range(timesteps_test)):
        t_inp = np.clip(
            np.ones(out_shape) * np.reshape(t / timesteps_test, (1, 1, 1, 1)),
            0,
            0.99999,
        )
        sch_params_t = schedule_model.predict([x, t_inp], verbose=0)
        alpha_t = np.clip(1 - sch_params_t[1] / timesteps_test, 1e-6, 0.99999)
        alpha_vec[..., t] = alpha_t
    gamma_vec = np.cumprod(alpha_vec, axis=-1)
    gamma_vec = np.clip(gamma_vec, 1e-10, 0.99999)
    count = 0
    pred_noise = 0
    for t in tqdm(range(timesteps_test, 1, -1)):
        z: Union[float, np.ndarray] = np.random.normal(0, 1, out_shape)
        if t == 1:
            z = 0
        alpha_t = alpha_vec[..., t - 1]
        beta_t = 1 - alpha_t
        gamma_t = gamma_vec[..., t - 1]
        gamma_tm1 = gamma_vec[..., t - 2]
        beta_factor = (1 - gamma_tm1) * beta_t / (1 - gamma_t)
        if count > 0:
            pred_sr = (
                np.sqrt(gamma_t) * pred_sr
                + np.sqrt(1 - gamma_t - beta_factor) * pred_noise
                + np.sqrt(beta_factor) * z
            )
        pred_noise = noise_model.predict([pred_sr, x, gamma_t], verbose=0)
        pred_sr = (pred_sr - np.sqrt(1 - gamma_t) * pred_noise) / np.sqrt(gamma_t)
        count += 1

    return pred_sr, gamma_vec, alpha_vec


def create_output_montage(
    pred_y: np.ndarray,
    gamma_vec: np.ndarray,
    y: np.ndarray,
    x: Optional[np.ndarray],
) -> np.ndarray:
    if pred_y.shape[3] > 1:
        channel_axis = 3
    else:
        channel_axis = None

    if x is not None:
        concatenated_images = np.concatenate(
            (pred_y, y, x, gamma_vec),
            axis=2,
        )
    else:
        concatenated_images = np.concatenate(
            (pred_y, y, gamma_vec),
            axis=2,
        )
    image: np.ndarray = montage(
        np.squeeze(concatenated_images),
        channel_axis=channel_axis,
    )
    return image


def log_loss(run: Optional[Run], avg_loss: np.ndarray, prefix: str) -> None:
    if run is not None:
        run[f"{prefix}_loss_sum"].log(avg_loss[0])
        run[f"{prefix}_loss_delta_noise"].log(avg_loss[1])
        run[f"{prefix}_loss_beta"].log(avg_loss[2])
        run[f"{prefix}_loss_KL"].log(avg_loss[3])
        run[f"{prefix}_loss_gamma"].log(avg_loss[4])


def log_metrics(
    run: Optional[Run], metrics_dict: Dict[str, float], prefix: str
) -> None:
    if run is not None:
        for metric_name, metric_value in metrics_dict.items():
            run[f"{prefix}_" + metric_name].log(metric_value)


def save_weighs(
    run: Optional[Run], model: Model, step: int, output_path: str, run_id: str
) -> None:

    model.save_weights(f"{output_path}/weights/model_{str(step)}_{run_id}.h5", True)

    if run is not None:
        run[f"model_weights/model_{str(step)}.h5"].upload(
            f"{output_path}/weights/model_{str(step)}_{run_id}.h5"
        )


def save_output_montage(
    run: Optional[Run],
    output_montage: np.ndarray,
    step: int,
    output_path: str,
    run_id: str,
    prefix: str,
    cmap: Optional[str] = None,
) -> None:

    plt.imsave(
        f"{output_path}/images/{prefix}_output_{str(step)}_{run_id}.png",
        output_montage,
        cmap=cmap,
    )

    if run is not None:
        run[f"{prefix}_images"].append(
            File(f"{output_path}/images/{prefix}_output_{str(step)}_{run_id}.png"),
            description=f"Step {step}, {prefix}",
        )


def obtain_output_montage_and_metrics(
    batch_x: np.ndarray,
    batch_y: np.ndarray,
    noise_model: Model,
    schedule_model: Model,
    generation_timesteps: int,
    task: str,
) -> Tuple[np.ndarray, Dict]:
    diff_inp = task in ["biosr_sr", "imagenet_sr"]

    pred_diff, gamma_vec, _ = ddpm_obtain_sr_img(
        batch_x,
        generation_timesteps,
        noise_model,
        schedule_model,
        batch_y.shape,
    )
    if diff_inp:
        pred_y = np.clip(pred_diff + batch_x, -1, 1)
    else:
        pred_y = np.clip(pred_diff, -1, 1)

    metrics = calculate_metrics(pred_y, batch_y)
    if task in ["biosr_sr", "imagenet_sr"]:
        gamma_vec = np.clip(gamma_vec[..., generation_timesteps // 2], -1, 1)
        montage_x = batch_x
    else:
        gamma_vec = np.clip(gamma_vec[..., 0:1, generation_timesteps // 2], -1, 1)
        montage_x = None
    output_montage = create_output_montage(
        pred_y,
        gamma_vec,
        batch_y,
        montage_x,
    )
    if task in ["biosr_sr", "imagenet_sr"]:
        output_montage = (output_montage * 127.5 + 127.5).astype(np.uint8)
    return output_montage, metrics
