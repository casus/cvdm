import tensorflow as tf
import numpy as np


def normal_kl(params):
    """
    Compute the KL divergence between two gaussians.
    KL divergence between normal distributions parameterized by mean and log-variance.

    """
    mean1, logvar1, mean2, logvar2 = params
    # return tf.zeros_like(mean1)
    kl_loss = 0.5 * (-1.0 + logvar2 - logvar1 + tf.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * tf.exp(-logvar2))
    safe_kl_loss = tf.where(tf.math.is_nan(kl_loss), tf.zeros_like(kl_loss), kl_loss)
    return safe_kl_loss


def uniform_gaussian_kl(params):
    a, b, mean1, logvar1 = params
    var_term = 0.5 * tf.math.log(2 * np.pi) + 0.5 * logvar1 - tf.math.log(b - a)
    mu_term = 1 / (b - a) * ((mean1 - a) ** 3 / (6 * tf.exp(logvar1)) - (mean1 - b) ** 3 / (6 * tf.exp(logvar1)))
    kl_loss = var_term + mu_term
    safe_kl_loss = tf.where(tf.math.is_nan(kl_loss), tf.zeros_like(kl_loss), kl_loss)

    return safe_kl_loss

def linear_loss(y_true, y_pred):
    return y_pred