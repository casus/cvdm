from typing import List
import tensorflow as tf


def normal_kl(data: List[float]) -> float:
    """
    Compute the KL divergence between two gaussians.
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    mean1, logvar1, mean2, logvar2 = data
    kl_loss: float = 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + tf.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * tf.exp(-logvar2)
    )
    return kl_loss


def linear_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return y_pred
