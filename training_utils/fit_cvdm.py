import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from diffusion_models.ddpm import train_model
from diffusion_losses.losses import linear_loss
from tqdm import tqdm
import os
from skimage.util import montage


def instantiate_cvdm(lr=1e-4, timesteps_train=2000, cond_shape=(256, 256, 3), out_shape=(256, 256, 3)):
    opt_m = tf.keras.optimizers.Adam(learning_rate=lr)
    p_model, t_model, sch_model = train_model(
        cond_shape, timesteps_train, out_shape[-1])
    t_model.compile(loss=linear_loss, loss_weights=[2, 1], optimizer=opt_m)
    return p_model, t_model, sch_model


def train_on_batch_cvdm(fx, fy, t_model, diff_inp=False):
    if diff_inp:
        dfy = fy -fx
    else:
        dfy = fy
    s_tdx = np.random.uniform(0, 1, (fx.shape[0], 1, 1, 1))
    ft = s_tdx * np.ones_like(dfy)
    loss = t_model.train_on_batch([dfy, fx, ft], np.zeros_like(dfy))
    loss_flow = np.array(loss)
    return loss_flow


def train_cvdm(XTrain, YTrain, train_steps=int(1e6),
               batch_size=3, lr=1e-4, timesteps_train=2000, timesteps_test=200, loss_rfreq=int(1e3),
               print_freq=int(1e3),
               training_outputs=True, checkpointing=True, training_outputs_path='.', dataset_name='srn'):
    opt_m = tf.keras.optimizers.Adam(learning_rate=lr)
    p_model, t_model, sch_model = train_model(
        (XTrain.shape[1], XTrain.shape[2], XTrain.shape[3]), timesteps_train, YTrain.shape[3])
    t_model.compile(loss=linear_loss, loss_weights=[2, 1], optimizer=opt_m)
    base_path = '.'

    os.makedirs(training_outputs_path, exist_ok=True)
    os.makedirs(training_outputs_path + '/schedules_' + str(dataset_name), exist_ok=True)
    os.makedirs(training_outputs_path + '/testing_' + str(dataset_name), exist_ok=True)
    os.makedirs(training_outputs_path + '/models_' + str(dataset_name), exist_ok=True)

    for step in range(train_steps):

        if step % loss_rfreq == 0:
            print('current learning rate: ' + str(t_model.optimizer.learning_rate.numpy()))
            avg_loss = 0

        r_idx = np.random.randint(0, XTrain.shape[0], batch_size)
        fx = XTrain[r_idx]
        fy = YTrain[r_idx]

        dfy = fy

        s_tdx = np.random.uniform(0, 1, (batch_size, 1, 1, 1))
        ft = s_tdx * np.ones_like(dfy)
        loss = t_model.train_on_batch([dfy, fx, ft], np.zeros_like(dfy))

        loss_flow = np.array(loss)
        avg_loss += 1 / (step % loss_rfreq + 1) * (loss_flow - avg_loss)
        print('step: ' + str(step) + 'loss cvdm: ' + str(avg_loss))
        if step % print_freq == 0 and training_outputs:
            pred_sr = np.random.normal(0, 1, fy.shape)
            # mean_pred = mu_model.predict(fx, verbose=False)

            # t_ft = np.expand_dims(np.expand_dims(ft, axis=-1), axis=-1) * np.ones_like(pred_sr)
            beta_schedule = []
            gamma_schedule = []
            for t in tqdm(range(timesteps_test, 1, -1)):
                z = np.random.normal(0, 1, fy.shape)
                if t == 1:
                    z = 0

                t_inp = np.clip(np.ones_like(ft) * np.reshape(t / timesteps_test, (1, 1, 1, 1)), 0, 0.99999)
                t_inp_1 = np.clip(np.ones_like(ft) * np.reshape((t - 1) / timesteps_test, (1, 1, 1, 1)), 0,
                                  0.99999)
                sch_params_t = sch_model.predict([fx, t_inp], verbose=False)
                sch_params_t_1 = sch_model.predict([fx, t_inp_1], verbose=False)
                pred_noise = p_model.predict([pred_sr, fx, sch_params_t[0]], verbose=False)

                alpha_p = np.clip(sch_params_t[0] / sch_params_t_1[0], 1e-6, 0.99999)
                beta_t = 1 - alpha_p
                beta_schedule.append(np.mean(beta_t[0, :, :, 0]))
                gamma_schedule.append(np.mean(sch_params_t[0][0, :, :, 0]))
                if t == timesteps_test // 2:
                    gamma_half = sch_params_t[0]
                alpha_factor = (beta_t) / np.sqrt(1 - sch_params_t[0])
                pred_sr = np.sqrt(1 / alpha_p) * (pred_sr - alpha_factor * pred_noise) + np.sqrt(beta_t) * z
                pred_sr = np.clip(pred_sr, -3, 3)

            # pred_sr = fx + pred_sr

            # pred_sr = pred_x0
            beta_schedule.reverse()
            gamma_schedule.reverse()

            plt.plot(beta_schedule)

            plt.savefig(f'{base_path}/schedules_{dataset_name}/schedule_beta_{str(step)}.png')
            plt.close('all')

            plt.plot(gamma_schedule)
            plt.savefig(f'{base_path}/schedules_{dataset_name}/schedule_gamma_{str(step)}.png')
            plt.close('all')

            int_res = np.clip(pred_sr, -1, 1)
            int_gt = fy
            int_x = fx

            c_imgs_result = montage(
                np.squeeze(np.concatenate((int_res, int_gt, int_x, np.clip(gamma_half, -1, 1)), axis=2)),
                multichannel=True)
            c_imgs_result = np.uint8(c_imgs_result * 127.5 + 127.5)
            # imageio.mimsave('./schedules/schedule_' + str(step) + '.gif', total_schedule)

            plt.imsave(f'{base_path}/testing_{dataset_name}/train_output_{str(step)}.png', c_imgs_result)

        if step % 20000 == 0 and checkpointing:  # 2000
            t_model.save_weights(f'{base_path}/models_{dataset_name}/model_{str(step)}.h5', True)
