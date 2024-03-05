from typing import Optional

import tensorflow as tf
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Concatenate, Conv2D, Dropout
from tensorflow_addons.layers import GroupNormalization

from cvdm.architectures.components.conditional_instance_normalization import (
    ConditionalInstanceNormalization,
)


def deep_resblock(
    x: tf.Tensor, noise_embedding: tf.Tensor, n_out_channels: int, dropout: float = 0.0
) -> tf.Tensor:
    """
    Reuse the resblock definitions from https://arxiv.org/pdf/1809.11096.pdf
    Slight tweak to use ConditionalInstanceNormalization and pre-activation
    as described in https://arxiv.org/pdf/1907.05600.pdf
    """
    assert n_out_channels is not None
    track_a = Conv2D(n_out_channels, 1, padding="same")(x)

    track_b = GroupNormalization(groups=32, axis=-1)(x)
    track_b = swish(track_b)
    track_b = Conv2D(n_out_channels, 1, padding="same")(track_b)
    track_b = ConditionalInstanceNormalization()([track_b, noise_embedding])
    track_b = swish(track_b)
    track_b = Conv2D(n_out_channels, 3, padding="same")(track_b)
    track_b = Dropout(dropout)(track_b)
    track_b = GroupNormalization(groups=32, axis=-1)(x)
    track_b = swish(track_b)
    track_b = Conv2D(n_out_channels, 3, padding="same")(track_b)
    track_b = ConditionalInstanceNormalization()([track_b, noise_embedding])
    track_b = swish(track_b)
    track_b = Conv2D(n_out_channels, 1, padding="same")(track_b)
    out: tf.Tensor = track_a + track_b
    return out


def up_deep_resblock(
    x: tf.Tensor,
    skip_x: tf.Tensor,
    noise_embedding: tf.Tensor,
    n_out_channels: int,
    dropout: float = 0.0,
) -> tf.Tensor:

    skip_x = skip_x / tf.math.sqrt(2.0)
    x = Concatenate(axis=-1)([x, skip_x])

    return deep_resblock(
        x, noise_embedding, n_out_channels=n_out_channels, dropout=dropout
    )
