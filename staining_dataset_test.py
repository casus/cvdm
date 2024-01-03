from training_utils import fit_cvdm, inf_cvdm
from benchmarks.staining_dataset import *
from tqdm import trange
import os
import matplotlib.pyplot as plt
from skimage.util import montage


def train_virus_staining_benchmark():
    dataset_name = 'hsv'
    print('loading Data')
    XTrain, YTrain = get_dataset(dataset_name)
    training_iter = int(1e6)
    checkpont_freq = int(2e4)
    print_freq = int(1e3)
    timesteps_test = int(2e2)
    p_size = 256
    ch = 1
    b_size = 3

    p_model, t_model, sch_model = fit_cvdm.instantiate_cvdm(cond_shape=(p_size, p_size, ch),
                                                            out_shape=(p_size, p_size, ch))
    training_outputs_path = '.'
    os.makedirs(training_outputs_path, exist_ok=True)
    os.makedirs(training_outputs_path + '/schedules_' + str(dataset_name), exist_ok=True)
    os.makedirs(training_outputs_path + '/testing_' + str(dataset_name), exist_ok=True)
    os.makedirs(training_outputs_path + '/models_' + str(dataset_name), exist_ok=True)

    with trange(training_iter) as t:
        for step in t:
            fx, fy = get_virus_signal_batch(XTrain, YTrain, p_size, b_size)
            loss = fit_cvdm.train_on_batch_cvdm(fx, fy, t_model, diff_inp=True)
            t.set_description(f'Step {str(step)}')
            t.set_postfix(cvdm_loss=loss[0], gamma_reg=loss[2], diff_loss=loss[1])

            if step % checkpont_freq == 0:
                t_model.save_weights(f'{training_outputs_path}/models_{dataset_name}/model_{str(step)}.h5', True)

            if step % print_freq == 0:
                pred_diff, gamma_vec, alpha_vec = inf_cvdm.ddpm_obtain_sr_img(fx, timesteps_test, p_model, sch_model)
                gamma_schedule = np.mean(gamma_vec, axis=(1, 2, 3))[0]
                beta_schedule = np.mean(1 - alpha_vec, axis=(1, 2, 3))[0]

                plt.plot(beta_schedule)

                plt.savefig(f'{training_outputs_path}/schedules_{dataset_name}/schedule_beta_{str(step)}.png')
                plt.close('all')

                plt.plot(gamma_schedule)
                plt.savefig(f'{training_outputs_path}/schedules_{dataset_name}/schedule_gamma_{str(step)}.png')
                plt.close('all')

                int_res = np.clip(pred_diff + fx, -1, 1)
                int_gt = fy
                int_x = fx

                c_imgs_result = montage(
                    np.squeeze(
                        np.concatenate((int_res, int_gt, int_x, np.clip(gamma_vec[..., timesteps_test // 2], -1, 1)),
                                       axis=2)))
                c_imgs_result = np.uint8(c_imgs_result * 127.5 + 127.5)

                plt.imsave(f'{training_outputs_path}/testing_{dataset_name}/train_output_{str(step)}.png',
                           c_imgs_result)


train_virus_staining_benchmark()
