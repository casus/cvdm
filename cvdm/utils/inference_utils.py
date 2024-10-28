import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from neptune import Run
from neptune.types import File
from skimage.util import montage
from tensorflow.keras.models import Model
from tqdm import tqdm




def ddpm_obtain_sr_img(
        x: np.ndarray,
        timesteps_test: int,
        noise_model: Model,
        schedule_model: Model,
        mu_model: Optional[Model],
        out_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if out_shape == None:
        out_shape = x.shape
    assert out_shape is not None
    pred_sr = np.random.normal(0, 1, out_shape)
    if mu_model is not None:
        mu_pred = mu_model.predict(x, verbose=0)[0]
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
        if mu_model is not None:
            pred_noise = noise_model.predict([pred_sr, x, mu_pred, gamma_t], verbose=0)
        else:
            pred_noise = noise_model.predict([pred_sr, x, gamma_t], verbose=0)
        pred_sr = (pred_sr - np.sqrt(1 - gamma_t) * pred_noise) / np.sqrt(gamma_t)
        count += 1
    if mu_model is not None:
        sigma = 0.5
        pred_diff = sigma * pred_sr + mu_pred
    else:
        pred_diff = pred_sr
    return pred_diff, gamma_vec, alpha_vec


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
    print(concatenated_images.shape)
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
        if len(avg_loss) == 6:
            run[f"{prefix}_loss_mean"].log(avg_loss[5])
    else:
        loss_labels = [
            "Loss Sum",
            "Delta Noise Loss",
            "Beta Loss",
            "KL Loss",
            "Gamma Loss",
        ]
        formatted_losses = [
            f"{label}: {loss:.6f}" for label, loss in zip(loss_labels, avg_loss[:5])
        ]
        for loss in formatted_losses:
            print(loss)
        if len(avg_loss) == 6:
            print(f"Mean Loss: {avg_loss[5]:.6f}")


def log_metrics(
        run: Optional[Run], metrics_dict: Dict[str, float], prefix: str
) -> None:
    if run is not None:
        for metric_name, metric_value in metrics_dict.items():
            run[f"{prefix}_" + metric_name].log(metric_value)
    else:
        print(f"{prefix.capitalize()} Metrics:")
        for metric_name, metric_value in metrics_dict.items():
            print(f"{metric_name}: {metric_value:.6f}")


def save_weights(
        run: Optional[Run],
        model: Model,
        mu_model: Optional[Model],
        step: int,
        output_path: str,
        run_id: str,
) -> None:
    weights_dir = f"{output_path}/weights"
    os.makedirs(weights_dir, exist_ok=True)

    model_weights_path = f"{weights_dir}/model_{str(step)}_{run_id}.h5"
    model.save_weights(model_weights_path)

    if run is not None:
        run[f"model_weights/model_{str(step)}.h5"].upload(model_weights_path)

    if mu_model is not None:
        mu_model_weights_path = f"{weights_dir}/mu_model_{str(step)}_{run_id}.h5"
        mu_model.save_weights(mu_model_weights_path)

        if run is not None:
            run[f"mu_model_weights/mu_model_{str(step)}.h5"].upload(
                mu_model_weights_path
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
    output_dir = f"{output_path}/images"
    os.makedirs(output_dir, exist_ok=True)

    image_path = f"{output_dir}/{prefix}_output_{str(step)}_{run_id}.png"
    plt.imsave(image_path, output_montage, cmap=cmap)

    if run is not None:
        run[f"{prefix}_images"].append(
            File(image_path),
            description=f"Step {step}, {prefix}",
        )


def obtain_output_montage_and_metrics(
        batch_x: np.ndarray,
        batch_y: np.ndarray,
        noise_model: Model,
        schedule_model: Model,
        mu_model: Optional[Model],
        generation_timesteps: int,
        diff_inp: bool,
        task: str,
) -> np.ndarray:
    pred_diff, gamma_vec, _ = ddpm_obtain_sr_img(
        batch_x,
        generation_timesteps,
        noise_model,
        schedule_model,
        mu_model,
        batch_y.shape,
    )
    if diff_inp:
        pred_y = pred_diff + batch_x
    else:
        pred_y = pred_diff

    if task == 'imagenet_sr':
        pred_y = np.clip(pred_y, -1, 1)
    else:
        pred_y = np.clip(pred_y, -1, 1)

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
    return output_montage
