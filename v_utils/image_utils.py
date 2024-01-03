import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + tf.math.tanh(tf.math.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = tf.math.log(tf.clip_by_value(cdf_plus, clip_value_min=1e-12))
    log_one_minus_cdf_min = tf.math.log(tf.clip_by_value((1.0 - cdf_min), clip_value_min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = tf.where(
        x < -0.999,
        log_cdf_plus,
        tf.where(x > 0.999, log_one_minus_cdf_min, tf.math.log(tf.clip_by_value(cdf_delta, clip_value_min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def rescale_image(image):
    # Rescale pixel values between 0 and 1
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # Rescale pixel values between -1 and 1
    return image


def sample_norm_01(x):
    x = np.float32(x)
    n_x = (x - np.amin(x, axis=(1, 2), keepdims=True)) / (
            np.amax(x, axis=(1, 2), keepdims=True) - np.amin(x, axis=(1, 2), keepdims=True))

    return n_x * 2 - 1


def plot_imgs(generated_imgs, gt, fx, label_images_folder, save_idx=-1, rescale=True):
    os.makedirs(label_images_folder + '/', exist_ok=True)
    for idx in range(generated_imgs.shape[0]):
        if save_idx == -1:
            s_idx = idx
        else:
            s_idx = idx + save_idx
        if rescale:
            int_pfy = prctile_norm(generated_imgs[idx, :, :, gt.shape[-1]], min_prc=0)
            int_gt = rescale_image(gt[idx, :, :, gt.shape[-1]])
            int_x = rescale_image(fx[idx, :, :, gt.shape[-1]])
        else:
            int_pfy = np.clip(generated_imgs[idx, :, :, gt.shape[-1]], -1, 1) * 0.5 + 0.5
            int_gt = np.clip(gt[idx, :, :, gt.shape[-1]], -1, 1) * 0.5 + 0.5
            int_x = np.clip(fx[idx, :, :, gt.shape[-1]], -1, 1) * 0.5 + 0.5
        c_imgs_result = np.concatenate((int_pfy, int_gt, int_x), axis=1)

        plt.imsave(label_images_folder + '/' + 'img_sr_' + str(s_idx) + '.png',
                   np.uint8(rescale_image(c_imgs_result) * 255.),
                   cmap='gray')
