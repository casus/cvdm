from typing import Tuple

import tensorflow as tf
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Input, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow_addons.layers import GroupNormalization

from cvdm.architectures.components.attention_block import attention_block
from cvdm.architectures.components.deep_residual_block import (
    deep_resblock,
    up_deep_resblock,
)
from cvdm.architectures.components.residual_block import resblock, up_resblock


def upsample(x: tf.Tensor, use_conv: bool = False) -> tf.Tensor:
    channels = x.shape[3]
    x = UpSampling2D(interpolation="nearest")(x)
    if use_conv:
        assert channels is not None
        x = Conv2D(channels, 3, padding="same")(x)
    return x


def downsample(x: tf.Tensor, use_conv: bool = False):
    channels = x.shape[3]
    if use_conv:
        assert channels is not None
        x = Conv2D(channels, 3, strides=2, padding="same")(x)
    else:
        x = AveragePooling2D(strides=2, padding="same")(x)
    return x


def sr3(
    img_shape: tf.TensorShape,
    gamma_shape: tf.TensorShape,
    channel_dim: int = 64,
    channel_ramp_multiplier: Tuple[int, ...] = (1, 2, 4, 8, 8),
    attention_resolution: Tuple[int, ...] = (8,),
    out_channels: int = 3,
    num_resblock: int = 3,
    dropout: float = 0,
    use_deep_blocks: bool = False,
    resample_with_conv: bool = False,
) -> Model:
    if use_deep_blocks:
        upblock = up_deep_resblock
        block = deep_resblock
    else:
        upblock = up_resblock
        block = resblock

    num_resolutions = len(channel_ramp_multiplier)
    combined_images = Input(img_shape)
    gamma_inp = Input(gamma_shape)
    noise_level_embedding = Conv2D(
        channel_dim, 7, activation="softplus", padding="same"
    )(gamma_inp)

    x = combined_images
    x = Conv2D(channel_dim, 3, padding="same")(x)
    skip_connections = [x]
    for i, multiplier in enumerate(channel_ramp_multiplier):
        for j in range(num_resblock):
            noise_level_embedding = Conv2D(channel_dim * multiplier, 1, padding="same")(
                noise_level_embedding
            )
            x = block(
                x, noise_level_embedding, channel_dim * multiplier, dropout=dropout
            )
            if x.shape[1] in attention_resolution:
                x = attention_block(x)
            skip_connections.append(x)
        if i != num_resolutions - 1:
            x = downsample(x, use_conv=resample_with_conv)
            noise_level_embedding = downsample(
                noise_level_embedding, use_conv=resample_with_conv
            )
            skip_connections.append(x)

    x = block(
        x=x,
        noise_embedding=noise_level_embedding,
        n_out_channels=x.shape[-1],
        dropout=dropout,
    )

    x = attention_block(x)

    x = block(
        x=x,
        noise_embedding=noise_level_embedding,
        n_out_channels=x.shape[-1],
        dropout=dropout,
    )

    for i, multiplier in reversed(list(enumerate(channel_ramp_multiplier))):
        for j in range(num_resblock + 1):

            noise_level_embedding = Conv2D(channel_dim * multiplier, 1, padding="same")(
                noise_level_embedding
            )
            x = upblock(
                x,
                skip_connections.pop(),
                noise_level_embedding,
                channel_dim * multiplier,
                dropout=dropout,
            )

            if x.shape[1] in attention_resolution:
                x = attention_block(x)
        if i != 0:
            x = upsample(x, use_conv=resample_with_conv)
            noise_level_embedding = upsample(
                noise_level_embedding, use_conv=resample_with_conv
            )

    assert len(skip_connections) == 0

    x = GroupNormalization(groups=32, axis=-1)(x)
    x = swish(x)
    outputs = Conv2D(out_channels, 3, padding="same")(x)

    model: Model = Model([combined_images, gamma_inp], outputs)
    return model
