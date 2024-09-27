from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from cvdm.configs.model_config import ModelConfig
from cvdm.diffusion_models.mean_model import mean_model
from cvdm.diffusion_models.noise_model import noise_model
from cvdm.diffusion_models.variance_model import variance_model
from cvdm.utils.data_utils import obtain_noisy_sample
from cvdm.utils.loss_utils import linear_loss, normal_kl
from cvdm.utils.training_utils import taylor_expand_gamma, time_grad


def create_joint_model(
    input_shape_condition: tf.TensorShape,
    timesteps: int,
    out_channels: int,
    model_config: ModelConfig,
) -> Tuple[Model, Model, Model, Optional[Model]]:
    """This function creates a Keras model for the image denoising pipeline."""

    ground_truth = Input(input_shape_condition[:-1] + (out_channels,))
    dirty_img = Input(input_shape_condition)

    timesteps_Lt = Input(input_shape_condition[:-1] + (out_channels,))
    timesteps_LT = tf.ones_like(timesteps_Lt) - (1 / timesteps)
    timesteps_L0 = tf.zeros_like(timesteps_Lt) + (1 / timesteps)

    sch_model = variance_model(input_shape_condition, out_channels)
    sch_params_LT = sch_model([dirty_img, timesteps_LT])
    sch_params_Lt = sch_model([dirty_img, timesteps_Lt])
    sch_params_L0 = sch_model([dirty_img, timesteps_L0])

    if model_config.zmd:
        pass

        sigma = 0.5
        n_model = noise_model(
            input_shape_condition,
            out_channels,
            model_type=model_config.noise_model_type,
            zmd=model_config.zmd,
        )
        mu_model = mean_model(input_shape_condition, out_channels)
        mean_pred = mu_model(dirty_img)[0]
        mean_pred_sg = tf.stop_gradient(mean_pred)
        n_sample_LT = Lambda(obtain_noisy_sample)(
            [(ground_truth - mean_pred_sg) / sigma, sch_params_LT[0]]
        )
        n_sample_Lt = Lambda(obtain_noisy_sample)(
            [(ground_truth - mean_pred_sg) / sigma, sch_params_Lt[0]]
        )
        pred_noise_Lt = n_model(
            [n_sample_Lt[0], dirty_img, mean_pred_sg, sch_params_Lt[0]]
        )

    else:
        n_model = noise_model(
            input_shape_condition,
            out_channels,
            model_type=model_config.noise_model_type,
        )

        n_sample_LT = Lambda(obtain_noisy_sample)([ground_truth, sch_params_LT[0]])
        n_sample_Lt = Lambda(obtain_noisy_sample)([ground_truth, sch_params_Lt[0]])
        pred_noise_Lt = n_model([n_sample_Lt[0], dirty_img, sch_params_Lt[0]])

    d_alpha_t = Lambda(time_grad)([sch_params_Lt[0], timesteps_Lt])

    gamma_exp = Lambda(
        taylor_expand_gamma, arguments={"n": model_config.snr_expansion_n}
    )(sch_params_Lt[0])
    d_alpha_n_t = Lambda(time_grad)([gamma_exp, timesteps_Lt])
    d2_alpha_n_t = Lambda(time_grad)([d_alpha_n_t, timesteps_Lt])

    kl_divergence_T = Lambda(normal_kl)(
        [
            tf.zeros_like(n_sample_LT[2]),
            tf.zeros_like(n_sample_LT[2]),
            n_sample_LT[2],
            tf.math.log(n_sample_LT[3] + 1e-7),
        ]
    )

    delta_noise = tf.square(pred_noise_Lt - n_sample_Lt[1])
    L_beta = (
        tf.square(d_alpha_t + sch_params_Lt[1] * sch_params_Lt[0])
        + tf.square(sch_params_L0[0] - 1)
        + tf.square(sch_params_LT[0])
    )
    L_gamma = model_config.alpha * tf.square(d2_alpha_n_t)
    joint_model: Model
    if model_config.zmd:
        delta_mean = tf.square(ground_truth - mean_pred)
        joint_model = Model(
            [ground_truth, dirty_img, timesteps_Lt],
            [delta_noise, L_beta, kl_divergence_T, L_gamma, delta_mean],
        )
        return n_model, joint_model, sch_model, mu_model
    else:
        joint_model = Model(
            [ground_truth, dirty_img, timesteps_Lt],
            [delta_noise, L_beta, kl_divergence_T, L_gamma],
        )
        joint_model.summary()
        return n_model, joint_model, sch_model, None


def instantiate_cvdm(
    lr: float,
    generation_timesteps: int,
    cond_shape: tf.TensorShape,
    out_shape: tf.TensorShape,
    model_config: ModelConfig,
) -> Tuple[Model, Model, Model, Optional[Model]]:
    opt_m = tf.keras.optimizers.Adam(learning_rate=lr)
    out_channels = out_shape[-1]
    assert out_channels is not None

    if model_config.zmd:
        models = create_joint_model(
            cond_shape, generation_timesteps, out_channels, model_config
        )
        noise_model, joint_model, schedule_model, mu_model = models
        joint_model.compile(loss=linear_loss, loss_weights=[1, 2, 2, 2, 2], optimizer=opt_m)  # type: ignore
    else:
        models = create_joint_model(
            cond_shape, generation_timesteps, out_channels, model_config
        )
        noise_model, joint_model, schedule_model, _ = models
        joint_model.compile(loss=linear_loss, loss_weights=[1, 2, 2, 2], optimizer=opt_m)  # type: ignore

    return models
