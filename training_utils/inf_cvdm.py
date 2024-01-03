import numpy as np
from tqdm import tqdm
from v_utils import image_utils


def ddpm_obtain_sr_img(fx, timesteps_test, p_model, sch_model, out_shape=None):
    if out_shape == None:
        out_shape = fx.shape
    pred_sr = np.random.normal(0, 1, out_shape)

    alpha_vec = np.zeros(out_shape + (timesteps_test,))
    for t in tqdm(range(timesteps_test)):
        t_inp = np.clip(np.ones(out_shape) * np.reshape(t / timesteps_test, (1, 1, 1, 1)), 0, 0.99999)
        sch_params_t = sch_model.predict([fx, t_inp], verbose=False)
        alpha_t = np.clip(1 - sch_params_t[1] / timesteps_test, 1e-6, 0.99999)
        alpha_vec[..., t] = alpha_t
    gamma_vec = np.cumprod(alpha_vec, axis=-1)

    for t in tqdm(range(timesteps_test, 1, -1)):
        z = np.random.normal(0, 1, out_shape)
        if t == 1:
            z = 0

        alpha_t = alpha_vec[..., t - 1]
        beta_t = 1 - alpha_t
        gamma_t = gamma_vec[..., t - 1]
        gamma_tm1 = gamma_vec[..., t - 2]
        pred_noise = p_model.predict([pred_sr, fx, gamma_t], verbose=False)

        beta_factor = (1 - gamma_tm1) * beta_t / (1 - gamma_t)
        alpha_factor = (beta_t) / np.sqrt(1 - gamma_t)
        if t > 1:
            pred_sr = np.sqrt(1 / alpha_t) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(beta_factor) * z
        else:
            pred_sr = (pred_sr - np.sqrt(1 - gamma_t) * pred_noise) / np.sqrt(gamma_t)

    pred_diff = pred_sr

    return pred_diff, gamma_vec, alpha_vec


def evaluate_test_set(XTest, YTest, models, batch_size=10, timesteps_test=200, n_samples=1, out_shape=None,
                      outputs_path='.', print_outputs=False):
    b_num = XTest.shape[0] // batch_size
    recon = np.zeros_like(YTest)
    p_model = models[0]
    sch_model = models[1]
    for idx in range(b_num):
        fx = XTest[idx * batch_size:(idx + 1) * batch_size]
        fy = YTest[idx * batch_size:(idx + 1) * batch_size]
        recon_samples = np.zeros((n_samples,) + fy.shape)
        print(recon_samples.shape)
        for jdx in range(n_samples):
            pred_dfy, gamma_vec, alpha_vec = ddpm_obtain_sr_img(fx, timesteps_test, p_model, sch_model,
                                                                out_shape=out_shape)
            recon_samples[jdx] = pred_dfy

        reconstruction = np.mean(recon_samples, axis=0)
        if print_outputs:
            image_utils.plot_imgs(reconstruction, fy, fx, outputs_path)

        recon[idx * batch_size:(idx + 1) * batch_size] = reconstruction
    return recon
