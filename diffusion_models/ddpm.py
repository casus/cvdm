from keras.layers import *
from tensorflow.keras.models import Model
from architectures.model import UNet_ddpm, sr3_model, time_network
from diffusion_losses.losses import normal_kl
from diffusion_models.ddpm_utils import *


def variance_model(image_shape, out_channels):
    blurry_image = Input(image_shape)
    t_input = Input(image_shape[:-1] + (out_channels,))
    t_model = time_network(image_shape[:-1] + (out_channels,), 1024, act='linear')
    t_int_model = time_network(image_shape[:-1] + (out_channels,), 1024, act='linear')
    t_clip = tf.clip_by_value(t_input, 0, 0.99999)
    model, _ = UNet_ddpm(blurry_image, inputs=blurry_image, gamma_inp=blurry_image,
                         out_filters=out_channels, z_enc=False, base_filters=32)
    model = Activation('softplus')(model)
    tau = t_model(t_clip)
    beta = model * tau
    gamma = tf.clip_by_value(tf.exp(-model * t_int_model(t_clip)), 1e-6, 0.99999)
    modified_model = Model(inputs=[blurry_image, t_input], outputs=[gamma, beta])

    return modified_model


def noise_predictor(input_shape_noisy, out_channels, model='sr3'):
    noisy_input = Input(input_shape_noisy[:-1] + (out_channels,))
    ref_frame = Input(input_shape_noisy)
    c_inpt = Concatenate()([noisy_input, ref_frame])
    gamma_inp = Input(input_shape_noisy[:-1] + (out_channels,))
    if model == 'sr3':
        s_model = sr3_model(input_shape_noisy[:-1] + (out_channels * 2,), input_shape_noisy[:-1] + (out_channels,),
                            out_channels=out_channels)

        noise_out = s_model([c_inpt, gamma_inp])
        model_out = Model([noisy_input, ref_frame, gamma_inp], noise_out)

    else:
        noise_out, _ = UNet_ddpm(input_shape_noisy, inputs=c_inpt, gamma_inp=gamma_inp,
                                 out_filters=out_channels, z_enc=False, base_filters=64)
        model_out = Model([noisy_input, ref_frame, gamma_inp], noise_out)

    return model_out


def train_model(input_shape_condition, timesteps, out_channels=1):
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

    n_model = noise_predictor(input_shape_condition, out_channels)

    n_sample_LT = Lambda(obtain_noisy_sample)([ground_truth, sch_params_LT[0]])
    n_sample_Lt = Lambda(obtain_noisy_sample)([ground_truth, sch_params_Lt[0]])
    pred_noise_Lt = n_model([n_sample_Lt[0], dirty_img, sch_params_Lt[0]])

    d_alpha_t = Lambda(time_grad)([sch_params_Lt[0], timesteps_Lt])
    d_beta_t = Lambda(time_grad)([sch_params_Lt[1], timesteps_Lt])
    d2_alpha_t = Lambda(time_grad)([d_alpha_t, timesteps_Lt])
    kl_divergence_T = Lambda(normal_kl)(
        [tf.zeros_like(n_sample_LT[2]), tf.zeros_like(n_sample_LT[2]), n_sample_LT[2],
         tf.math.log(n_sample_LT[3] + 1e-7)])
    delta_noise = tf.square(pred_noise_Lt - n_sample_Lt[1])
    diff_x0 = kl_divergence_T + (
            tf.square(d_alpha_t + sch_params_Lt[1] * sch_params_Lt[0]) + tf.square(
        sch_params_L0[0] - 1) + tf.square(sch_params_LT[0]) + (1e-3) * tf.square(d2_alpha_t))
    train_model = Model([ground_truth, dirty_img, timesteps_Lt],
                        [diff_x0, delta_noise])
    train_model.summary()
    return n_model, train_model, sch_model
