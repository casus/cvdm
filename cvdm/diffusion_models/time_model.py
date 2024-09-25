import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Input
from tensorflow.keras.models import Model


def time_model(
    t_shape: tf.TensorShape, n_hidden: int, act: str = "softplus", mult: bool = True
) -> Model:
    t_input = Input(t_shape)
    l1 = Conv2D(
        1,
        kernel_size=(1, 1),
        padding="same",
        kernel_constraint=tf.keras.constraints.NonNeg(),
    )(t_input)
    x = Conv2D(
        n_hidden,
        kernel_size=(3, 3),
        padding="same",
        activation="sigmoid",
        kernel_constraint=tf.keras.constraints.NonNeg(),
    )(l1)
    x = Conv2D(
        1,
        kernel_size=(1, 1),
        padding="same",
        kernel_constraint=tf.keras.constraints.NonNeg(),
    )(x)
    if mult:
        out = Activation(act)(x + l1) * t_input
    else:
        out = Activation(act)(x + l1)
    model: Model = Model(t_input, out)
    return model
