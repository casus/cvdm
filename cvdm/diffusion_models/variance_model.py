import contextlib
from typing import Any, Dict, Iterator
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input

from cvdm.architectures.unet import UNet
from cvdm.diffusion_models.time_model import time_model


@contextlib.contextmanager
def options(options: Dict[str, Any]) -> Iterator[None]:
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


def variance_model(image_shape: tf.TensorShape, out_channels: int) -> Model:
    blurry_image = Input(image_shape)
    t_input = Input(image_shape[:-1] + (out_channels,))
    t_model = time_model(image_shape[:-1] + (out_channels,), 1024, act="linear")
    t_int_model = time_model(image_shape[:-1] + (out_channels,), 1024, act="linear")
    with options({"layout_optimizer": False}):
        t_clip = tf.clip_by_value(t_input, 0, 0.99999)
    model, _ = UNet(
        blurry_image,
        inputs=blurry_image,
        gamma_inp=blurry_image,
        out_filters=out_channels,
        base_filters=32,
    )
    model = Activation("softplus")(model)
    tau = t_model(t_clip)
    beta = model * tau
    gamma = tf.clip_by_value(tf.exp(-model * t_int_model(t_clip)), 1e-6, 0.99999)
    modified_model: Model = Model(inputs=[blurry_image, t_input], outputs=[gamma, beta])

    return modified_model
