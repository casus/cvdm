import tensorflow as tf


def time_grad(x):
    gamma_t = x[0]
    timestep = x[1]

    d_gamma = tf.gradients(gamma_t, timestep, unconnected_gradients=tf.UnconnectedGradients.ZERO)[0]
    return d_gamma


def obtain_noisy_sample(x):
    x_0 = x[0]
    gamma = x[1]

    noise_sample = tf.random.normal(tf.shape(x_0))

    safe_sample = tf.sqrt(gamma) * x_0 + tf.sqrt(1 - gamma) * noise_sample
    safe_mean = tf.sqrt(gamma) * x_0
    safe_sample = tf.where(tf.math.is_nan(safe_sample), tf.zeros_like(safe_sample), safe_sample)
    safe_mean = tf.where(tf.math.is_nan(safe_mean), tf.zeros_like(safe_mean), safe_mean)
    return [safe_sample, noise_sample, safe_mean, 1 - gamma]


def obtain_residuals(x):
    sigma = 2
    x_t = x[0]
    mu_context = x[1]
    gamma_vec = x[2]
    noise_sample = tf.random.normal(tf.shape(x_t))
    clean_residual = (x_t - mu_context) / sigma
    return [tf.sqrt(gamma_vec) * clean_residual + tf.sqrt(1 - gamma_vec) * noise_sample, noise_sample]
