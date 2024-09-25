from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization


def tile_gamma(x: np.ndarray) -> tf.Tensor:
    gamma = x[0]
    noisy_y = x[1]
    tile_gamma: tf.Tensor = tf.ones_like(noisy_y) * tf.expand_dims(
        tf.expand_dims(gamma, axis=-1), axis=-1
    )
    return tile_gamma


def UNet(
    shape: tf.TensorShape,
    inputs: Optional[Input] = None,
    gamma_inp: Optional[Input] = None,
    out_filters: int = 2,
    base_filters: int = 64,
    full_model=False,
) -> Union[Tuple[tf.Tensor, tf.Tensor], Model]:

    if inputs is None:
        inputs = Input(shape)

    conv1 = Conv2D(
        base_filters,
        3,
        activation="softplus",
        padding="same",
    )(inputs)

    if gamma_inp is not None:
        conv_gamma = Conv2D(
            base_filters,
            7,
            activation="softplus",
            padding="same",
        )(gamma_inp)
        conv1 = Add()([conv_gamma, conv1])

    conv1 = InstanceNormalization()(conv1)
    conv1 = Conv2D(
        base_filters,
        3,
        activation="softplus",
        padding="same",
    )(conv1)
    conv1 = InstanceNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = (
        Conv2D(
            base_filters * 2,
            3,
            activation="softplus",
            padding="same",
        )
    )(pool1)
    conv2 = InstanceNormalization()(conv2)
    conv2 = Conv2D(
        base_filters * 2,
        3,
        activation="softplus",
        padding="same",
    )(conv2)
    conv2 = InstanceNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(
        base_filters * 4,
        3,
        activation="softplus",
        padding="same",
    )(pool2)
    conv3 = InstanceNormalization()(conv3)
    conv3 = Conv2D(
        base_filters * 4,
        3,
        activation="softplus",
        padding="same",
    )(conv3)
    conv3 = InstanceNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(
        base_filters * 8,
        3,
        activation="softplus",
        padding="same",
    )(pool3)
    conv4 = InstanceNormalization()(conv4)
    conv4 = Conv2D(
        base_filters * 16,
        3,
        activation="softplus",
        padding="same",
    )(conv4)
    conv4 = InstanceNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(
        base_filters * 16,
        3,
        activation="softplus",
        padding="same",
    )(pool4)
    conv5 = InstanceNormalization()(conv5)
    conv5 = Conv2D(
        base_filters * 16,
        3,
        activation="softplus",
        padding="same",
    )(conv5)
    conv5 = InstanceNormalization()(conv5)

    up6 = Conv2D(base_filters * 16, 2, activation="softplus", padding="same")(
        Conv2DTranspose(
            512,
            3,
            activation="softplus",
            padding="same",
            strides=2,
        )(conv5)
    )

    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(
        base_filters * 8,
        3,
        activation="softplus",
        padding="same",
    )(merge6)
    conv6 = InstanceNormalization()(conv6)
    conv6 = Conv2D(
        base_filters * 8,
        3,
        activation="softplus",
        padding="same",
    )(conv6)
    conv6 = InstanceNormalization()(conv6)
    up7 = Conv2D(base_filters * 4, 2, activation="softplus", padding="same")(
        Conv2DTranspose(
            256,
            3,
            activation="softplus",
            padding="same",
            strides=2,
        )(conv6)
    )
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(
        base_filters * 4,
        3,
        activation="softplus",
        padding="same",
    )(merge7)
    conv7 = InstanceNormalization()(conv7)
    conv7 = Conv2D(
        base_filters * 4,
        3,
        activation="softplus",
        padding="same",
    )(conv7)
    conv7 = InstanceNormalization()(conv7)
    up8 = Conv2D(base_filters * 2, 2, activation="softplus", padding="same")(
        Conv2DTranspose(
            128,
            3,
            activation="softplus",
            padding="same",
            strides=2,
        )(conv7)
    )
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(
        base_filters * 2,
        3,
        activation="softplus",
        padding="same",
    )(merge8)
    conv8 = InstanceNormalization()(conv8)
    conv8 = Conv2D(
        base_filters * 2,
        3,
        activation="softplus",
        padding="same",
    )(conv8)
    conv8 = InstanceNormalization()(conv8)

    up9 = Conv2D(base_filters, 2, activation="softplus", padding="same")(
        Conv2DTranspose(
            64,
            3,
            activation="softplus",
            padding="same",
            strides=2,
        )(conv8)
    )

    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(
        base_filters,
        3,
        activation="softplus",
        padding="same",
    )(merge9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv2D(
        base_filters,
        3,
        activation="softplus",
        padding="same",
    )(conv9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv2D(
        base_filters,
        3,
        activation="softplus",
        padding="same",
    )(conv9)
    conv9 = InstanceNormalization()(conv9)
    conv10 = Conv2D(
        out_filters,
        1,
        activation="linear",
        padding="same",
    )(conv9)

    if full_model and gamma_inp is not None:
        model: Model = Model([inputs, gamma_inp], conv10)
        return model
    elif full_model:
        model = Model(inputs, conv10)
        return model
    else:
        return conv10, conv5
